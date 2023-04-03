import glob
import itertools
import warnings

import dcpy
import numpy as np
import pandas as pd

import xarray as xr

from . import OPTIONS, mdjwf, mixpods
from .constants import section_lons  # noqa

TAO_STANDARD_NAMES = {
    "WU_422": "eastward_wind",
    "WV_423": "northward_wind",
    "RH_910": "relative_humidity",
    "T_25": "sea_surface_temperature",
    "AT_21": "air_temperature",
    "WS_401": "wind_speed",
    "WD_410": "wind_from_direction",
    "TX_442": "surface_downward_eastward_stress",
    "TY_443": "surface_downward_northward_stress",
    # "TAU_440": "",
}

ROOT = OPTIONS["root"]


def read_all(domain=None):
    johnson = read_johnson()
    tao = read_tao(domain)
    sst = read_sst(domain)
    oscar = read_oscar(domain)

    return [johnson, tao, sst, oscar]


def read_johnson(filename=None):
    if filename is None:
        filename = f"{ROOT}/obs/johnson-eq-pac-adcp.cdf"
    ds = xr.open_dataset(filename).rename(
        {
            "XLON": "longitude",
            "YLAT11_101": "latitude",
            "ZDEP1_50": "depth",
            "POTEMPM": "temp",
            "SALINITYM": "salt",
            "SIGMAM": "dens",
            "UM": "u",
            "XLONedges": "lon_edges",
        }
    )

    ds["dens"] += 1000
    ds["longitude"] = ds.longitude - 360
    ds["depth"] = ds.depth * -1
    ds["depth"].attrs["units"] = "m"
    ds["u"].attrs["units"] = "m/s"

    return ds


def read_tao_adcp(domain=None, freq="dy", dirname=None):
    if dirname is None:
        dirname = ROOT + "/obs/tao/"

    if freq == "dy":
        adcp = xr.open_dataset(f"{dirname}/adcp_xyzt_dy.cdf").rename(
            {"lon": "longitude", "lat": "latitude", "U_1205": "u", "V_1206": "v"}
        )
    elif freq == "hr":
        adcp = tao_read_and_merge("hr", "adcp")

    adcp = adcp.chunk({"latitude": 1, "longitude": 1})

    for vv in adcp:
        adcp[vv] /= 100
        adcp[vv].attrs["long_name"] = vv
        adcp[vv].attrs["units"] = "m/s"
        adcp[vv] = adcp[vv].where(np.abs(adcp[vv]) < 1000)

    adcp["u"].attrs["units"] = "m/s"
    adcp["v"].attrs["units"] = "m/s"

    if domain is not None:
        adcp = adcp.sel(**domain)
    else:
        adcp = adcp.sel(latitude=0)

    return adcp  # .dropna("longitude", how="all").dropna("depth", how="all")


def tao_read_and_merge(suffix, kind):
    """read non-ADCP files."""
    if kind == "temp":
        prefix = "t"
        renamer = {"T_20": "T"}

    elif kind == "salt":
        prefix = "s"
        renamer = {"S_41": "S"}

    elif kind == "cur":
        prefix = "cur"
        renamer = {"U_320": "u", "V_321": "v"}

    elif kind == "adcp":
        prefix = "adcp"
        renamer = {"u_1205": "u", "v_1206": "v"}

    elif kind == "met":
        prefix = "met"
        renamer = {
            "WU_422": "uwnd",
            "WV_423": "vwnd",
            "RH_910": "relhum",
            "T_25": "sst",
            "AT_21": "airt",
            "WS_401": "wndspd",
            "WD_410": "wnddir",
        }

    elif kind == "tau":
        prefix = "tau"
        renamer = {
            "TX_442": "taux",
            "TY_443": "tauy",
            "TAU_440": "tau",
            "TD_445": "taudir",
        }
    elif kind == "rad":
        prefix = "*nc"
        renamer = {
            "SWN_1495": "swnet",
            "LWN_1136": "lwnet",
            "QS_138": "qsen",
            "QL_137": "qlat",
            "QT_210": "qnet",
        }

    ds = []

    tfiles = tuple(
        itertools.chain(
            *[
                glob.glob(f"{ROOT}/obs/tao/{prefix}0n{lon}_{suffix}.cdf")
                for lon in ["156e", "165e", "170w", "140w", "110w"]
            ]
        )
    )

    if not tfiles:
        raise ValueError(f"0 files were found: {tfiles}")

    ds = xr.open_mfdataset(tfiles, chunks={"time": 1000000}, join="outer")
    if renamer:
        ds = ds[list(renamer.keys())]
    for name in set(TAO_STANDARD_NAMES) & set(ds.variables):
        ds[name].attrs["standard_name"] = TAO_STANDARD_NAMES[name]

    merged = ds.rename({"lon": "longitude", "lat": "latitude"}).cf.guess_coord_axis(
        verbose=True
    )

    for var in merged:
        merged[var] = merged[var].where(merged[var] < 1e9)

    with xr.set_options(keep_attrs=True):
        merged["longitude"] = merged.longitude - 360
        if "depth" in merged:
            merged["depth"] = -1 * merged.depth
            merged["depth"].attrs.update({"units": "m", "axis": "Z", "positive": "up"})
    return merged.rename_vars(renamer)


def tao_merge_10m_and_hourly(kind):
    """Merge 10minute and hourly data into one record."""
    m10 = tao_read_and_merge("10m", kind)
    hr = tao_read_and_merge("hr", kind)

    # instead reindex to a 10min freq and use coarsen.
    new_index = pd.date_range(
        start=m10.time[0].dt.round("H").values,
        end=m10.time[-1].dt.round("H").values,
        freq="10min",
    )
    m10 = m10.reindex(time=new_index, method="nearest")
    m10hr = m10.chunk({"longitude": 1}).coarsen(time=6, boundary="trim").mean()

    new_hourly_index = pd.date_range(
        start=np.min([m10.time[0].values, hr.time[0].values]),
        end=np.max([m10.time[-1].values, hr.time[-1].values]),
        freq="H",
    )

    concat = xr.concat(
        [
            m10hr.reindex(time=new_hourly_index, method="nearest"),
            hr.reindex(time=new_hourly_index, method="nearest"),
        ],
        dim="concat",
    )

    # # T at 170W is only available at 10min frequency
    # ds.append(xr.load_dataset(ROOT+'/obs/tao/t0n170w_10m.cdf')['T_20']
    #           .resample(time='H').mean('time'))

    # adcp = read_tao_adcp(freq='hr')
    # return xr.merge(ds)
    result = concat.mean("concat").squeeze()
    result.attrs = hr.attrs
    for variable in hr.variables:
        result[variable].attrs = hr[variable].attrs

    return result


def read_eq_tao_cur_hr():
    return tao_merge_10m_and_hourly("cur") / 100


def read_eq_tao_salt_hr():
    return tao_read_and_merge("hr", "salt").S


def read_eq_tao_temp_hr():
    """Read hourly resolution temperature for equatorial moorings."""

    # sfiles = [ROOT+'/obs/tao/s0n'+lon+'_hr.cdf'
    #           for lon in ['156e', '165e', '140w', '110w', '170w']]
    # for file in tqdm.tqdm(sfiles):
    #     ds.append(xr.open_dataset(file)['S_41'])

    return tao_merge_10m_and_hourly("temp").T


def read_tao(domain=None):
    tao = xr.open_mfdataset(
        [
            ROOT + "/obs/tao/" + ff
            for ff in ["t_xyzt_dy.cdf", "s_xyzt_dy.cdf", "cur_xyzt_dy.cdf"]
        ],
        parallel=False,
        chunks={"lat": 1, "lon": 1, "depth": 5},
    ).rename(
        {
            "U_320": "u",
            "V_321": "v",
            "T_20": "temp",
            "S_41": "salt",
            "lon": "longitude",
            "lat": "latitude",
        }
    )
    tao["longitude"] -= 360

    tao["u"] /= 100
    tao["v"] /= 100
    tao["u"].attrs["units"] = "m/s"
    tao["v"].attrs["units"] = "m/s"

    tao["depth"] *= -1

    tao = tao.drop(
        [
            "S_300",
            "D_310",
            "QS_5300",
            "QD_5310",
            "QS_5041",
            "SS_6041",
            "QT_5020",
            "ST_6020",
            "SRC_6300",
        ]
    )

    for vv in tao:
        tao[vv] = tao[vv].where(tao[vv] < 1e4)
        tao[vv].attrs["long_name"] = ""

    tao["dens"] = mdjwf.dens(tao.salt, tao.temp, tao.depth)

    if domain is not None:
        return tao.sel(**domain)
    else:
        return tao


def read_sst(domain=None):
    if domain is not None:
        years = range(
            pd.Timestamp(domain["time"].start).year,
            pd.Timestamp(domain["time"].stop).year,
        )
        sst = xr.open_mfdataset(
            [ROOT + "/obs/oisst/sst.day.mean." + str(yy) + ".nc" for yy in years],
            parallel=True,
        )
    else:
        sst = xr.open_mfdataset(ROOT + "/obs/oisst/sst.day.mean.*.nc", parallel=True)

    sst["lon"] -= 360

    sst = sst.rename({"lat": "latitude", "lon": "longitude"})

    sst["anom"] = sst.sst - sst.sst.mean(["longitude", "time"])
    sst["anom"].attrs["long_name"] = "OISST Anomaly"
    sst["anom"].attrs["units"] = r"$\degree$C"
    sst["anom"].attrs["description"] = (
        "SST - mean(SST) in longitude, "
        "time after subsetting to simulation "
        "time length"
    )

    if domain is not None:
        return sst.sel(**domain)
    else:
        return sst


def read_oscar(domain=None):
    oscar = dcpy.oceans.read_oscar(ROOT + "/obs/oscar/").rename(
        {"lat": "latitude", "lon": "longitude"}
    )
    oscar["longitude"] = oscar["longitude"] - 360
    oscar = oscar.sortby("latitude")

    if domain is not None:
        return oscar.sel(**domain)
    else:
        return oscar


def read_argo():
    dirname = ROOT + "/obs/argo/"
    chunks = {"LATITUDE": 1, "LONGITUDE": 1}

    argoT = xr.open_dataset(
        dirname + "RG_ArgoClim_Temperature_2017.nc", decode_times=False, chunks=chunks
    )
    argoS = xr.open_dataset(
        dirname + "RG_ArgoClim_Salinity_2017.nc", decode_times=False, chunks=chunks
    )

    argoS["S"] = argoS.ARGO_SALINITY_ANOMALY + argoS.ARGO_SALINITY_MEAN
    argoT["T"] = argoT.ARGO_TEMPERATURE_ANOMALY + argoT.ARGO_TEMPERATURE_MEAN

    argo = argoT.update(argoS)

    argo = argo.rename(
        {
            "LATITUDE": "latitude",
            "LONGITUDE": "longitude",
            "PRESSURE": "depth",
            "TIME": "time",
            "ARGO_TEMPERATURE_MEAN": "Tmean",
            "ARGO_TEMPERATURE_ANOMALY": "Tanom",
            "ARGO_SALINITY_MEAN": "Smean",
            "ARGO_SALINITY_ANOMALY": "Sanom",
        }
    )

    _, ref_date = xr.coding.times._unpack_netcdf_time_units(argo.time.attrs["units"])

    argo.time.values = pd.Timestamp(ref_date) + pd.to_timedelta(
        30 * argo.time.values, unit="D"
    )
    argo["longitude"] -= 360
    argo["depth"] *= -1

    return argo


def process_nino34():
    nino34 = process_esrl_index("nino34.data", skipfooter=5)
    return nino34  # nino34.to_netcdf(ROOT + "/obs/nino34.nc")


def process_oni():
    oni = process_esrl_index("oni.data", skipfooter=8)
    return oni
    # oni.to_netcdf(ROOT + "/obs/oni.nc")


def process_esrl_index(file, skipfooter=3):
    """Read and make xarray version of climate indices from ESRL."""
    month_names = (
        pd.date_range("01-Jan-2001", "31-Dec-2001", freq="MS")
        .to_series()
        .dt.strftime("%b")
        .values.astype(str)
    )

    index = pd.read_csv(
        ROOT + "/obs/" + file,
        index_col=0,
        names=month_names,
        delim_whitespace=True,
        skiprows=1,
        na_filter=False,
        skipfooter=skipfooter,
        engine="python",
        dtype=np.float32,
    )

    flat = index.stack().reset_index()
    flat["time"] = pd.date_range(
        "01-Jan-" + str(flat["level_0"].iloc[0]),
        "02-Dec-" + str(flat["level_0"].iloc[-1]),
        freq="MS",
    )
    da = (
        flat.drop(["level_0", "level_1"], axis=1)
        .rename({0: "index"}, axis=1)
        .set_index("time")
        .to_xarray()
    )

    return da.where(da > -90)["index"]


def read_jra(files=None, chunks={"time": 1200}, correct_time=False):
    if files is None:
        raise ValueError("files cannot be none. this is now a helper function.")

    jra = xr.open_mfdataset(
        files, combine="by_coords", decode_times=False, chunks=chunks, parallel=True
    ).rename({"lat": "latitude", "lon": "longitude"})

    if correct_time:
        raise ValueError("This is definitely junk")
        jra["time"] = jra.time - jra.time[0]
        jra.time.attrs["units"] = "days since 1995-09-01"

    else:
        jra.time.attrs["units"] = "days since 1900-01-01"

    jra["longitude"] = jra.longitude - 360

    jra = jra.sel(longitude=slice(-170, -95), latitude=slice(-12, 12))

    renamer = {
        "Uwind": "uas",
        "Vwind": "vas",
        "Tair": "tas",
        "Pair": "psl",
        "Qair": "huss",
        "lwrad_down": "rlds",
        "swrad": "rsds",
        "rain": "prra",
    }

    if any([key in jra for key in renamer.keys()]):
        jra = jra.rename({k: v for k, v in renamer.items() if k in jra})

    return xr.decode_cf(jra)


def read_jra_95():
    jradir = "/glade/work/dcherian/pump/combined_95_97_JRA/"

    jrafull = read_jra(
        [
            f"{jradir}/JRA55DO_1.3_Tair.nc",
            f"{jradir}/JRA55DO_1.3_Uwind.nc",
            f"{jradir}/JRA55DO_1.3_Qair.nc",
            f"{jradir}/JRA55DO_1.3_Vwind.nc",
            f"{jradir}/JRA55DO_1.3_Pair.nc",
        ],
        chunks={"time": 120},
        correct_time=False,
    )

    jrafull2 = read_jra(
        [
            f"{jradir}/JRA55DO_1.3_rain.nc",
            f"{jradir}/JRA55DO_1.3_lwrad_down.nc",
            f"{jradir}/JRA55DO_1.3_swrad.nc",
        ],
        chunks={"time": 120},
        correct_time=False,
    )
    jrafull2["time"] = jrafull2.time - pd.Timedelta("1.5h")
    assert jrafull.indexes["time"].equals(jrafull2.indexes["time"])

    jrafull = jrafull.merge(jrafull2)
    # should be 7h BUT there is a three hour offset
    jrafull["time"] = jrafull.time - pd.Timedelta("4h")

    return jrafull


def read_jra_20():
    jradir = "/glade/campaign/cgd/oce/people/bachman/make_TPOS_MITgcm/1_20_1999-2018/JRA_FORCING"

    # the time offset for 1999 is different from the rest!
    jrafull = xr.concat(
        [
            # read_jra(
            #    [
            #        f"{jradir}/1999/JRA55DO_1.3_Tair_1999.nc",
            #        f"{jradir}/1999/JRA55DO_1.3_Uwind_1999.nc",
            #        f"{jradir}/1999/JRA55DO_1.3_Qair_1999.nc",
            #        f"{jradir}/1999/JRA55DO_1.3_Vwind_1999.nc",
            #        f"{jradir}/1999/JRA55DO_1.3_Pair_1999.nc",
            #    ],
            #    chunks={"time": 1200},
            # ),
            read_jra(f"{jradir}/20[0-9][0-9]/JRA55DO*[0-9].nc", chunks={"time": 1200}),
        ],
        dim="time",
    )
    # print(jrafull.time)

    jrafull2 = xr.concat(
        [
            # read_jra(
            #    [
            #        f"{jradir}/1999/JRA55DO_1.3_rain_1999.nc",
            #        f"{jradir}/1999/JRA55DO_1.3_lwrad_down_1999.nc",
            #        f"{jradir}/1999/JRA55DO_1.3_swrad_1999.nc",
            #    ],
            #    chunks={"time": 1200},
            # ),
            read_jra(f"{jradir}/20[0-9][0-9]/JRA55DO*[0-9].nc", chunks={"time": 1200}),
        ],
        dim="time",
    )
    # jrafull2["time"] = jrafull2.time - pd.Timedelta("1.5h")

    # print(jrafull2.time)
    # xr.align(jrafull, jrafull2, join="exact")

    # jrafull2["time"] = jrafull.time.data
    jrafull = jrafull.merge(jrafull2)
    jrafull["time"] = jrafull.time - pd.Timedelta("7h")

    return jrafull


def interp_jra_to_station(jrafull, station):
    jra = jrafull.sel(
        time=slice(station.time[0].values, station.time[-1].values)
    ).interp(longitude=station.longitude.values, latitude=station.latitude.values)

    jra = dcpy.dask.map_copy(jra)

    jrai = jra.compute().interp(time=station.time)
    return jrai


def read_drifters(kind="annual"):
    if kind == "annual":
        drifter = xr.load_dataset("~/pump/glade/obs/drifter/drifter_annualmeans.nc")

    drifter = drifter.rename({"Lon": "longitude", "Lat": "latitude"}).roll(
        longitude=720, roll_coords=True
    )

    drifter["longitude"] = (
        xr.where(drifter.longitude < 0, drifter.longitude + 360, drifter.longitude)
        - 360
    )
    return drifter.sel(longitude=slice(-230, -90))


def read_tao_zarr(kind="gridded", **kwargs):
    if kind not in ["gridded", "merged", "ancillary"]:
        raise ValueError(
            f"'kind' must be one of ['gridded', 'merged']. Received {kind!r}"
        )

    root = ROOT + "/zarrs/"
    if kind == "merged":
        tao = xr.open_zarr(f"{root}/tao_eq_hr_merged_cur.zarr", **kwargs)
    elif kind == "gridded":
        tao = xr.open_zarr(f"{root}/tao_eq_hr_gridded.zarr", **kwargs)
    elif kind == "ancillary":
        tao = xr.open_zarr(f"{root}/tao-gridded-ancillary.zarr", **kwargs)

    orig = tao.copy()
    # tao.depth.attrs.update({"axis": "Z", "positive": "up"})
    # tao = tao.chunk({"depth": -1, "time": 10000})
    tao["densT"] = dcpy.eos.pden(35 * xr.ones_like(tao.T), tao.T, tao.depth)
    tao.densT.attrs["long_name"] = "$ρ_T$"
    tao.densT.attrs["description"] = "density from T only, assuming S=35"

    tao["dens"] = dcpy.eos.pden(tao.S, tao.T, tao.depth)
    tao.densT.attrs["description"] = "density using T, S"

    tao["theta"] = dcpy.eos.ptmp(35 * xr.ones_like(tao.T), tao.T, tao.depth)
    tao.theta.attrs["description"] = "potential temperature using T, S=35"

    for var in orig.variables:
        tao[var].attrs = orig[var].attrs

    return tao


def get_nan_block_lengths(obj, dim, index):
    """
    Return an object where each NaN element in 'obj' is replaced by the
    length of the gap the element is in.
    """

    from xarray import Variable, ones_like

    # make variable so that we get broadcasting for free
    index = Variable([dim], index)

    # algorithm from https://github.com/pydata/xarray/pull/3302#discussion_r324707072
    arange = ones_like(obj) * index
    valid = obj.notnull()
    valid_arange = arange.where(valid)
    cumulative_nans = valid_arange.ffill(dim=dim).fillna(index[0])

    nan_block_lengths = (
        cumulative_nans.diff(dim=dim, label="upper")
        .reindex({dim: obj[dim]})
        .where(valid)
        .bfill(dim=dim)
        .where(~valid, 0)
        .fillna(index[-1] - valid_arange.max())
    )

    return nan_block_lengths


def make_enso_mask_old(threshold=6):
    oni = process_oni()
    ntime = oni.sizes["time"]
    # freq = xr.infer_freq(oni.time)
    fill_value = "_______"

    enso = xr.DataArray(
        np.full((ntime,), fill_value=fill_value),
        dims=("time"),
        coords={"time": oni.time},
        name="ENSO",
    )

    # threshold =  6 #int(pd.Timedelta(threshold) / pd.Timedelta(f"1{freq}"))
    for phase, mask in zip(["El-Nino", "La-Nina"], [oni >= 0.5, oni <= -0.5]):
        length = get_nan_block_lengths(
            xr.where(mask, np.nan, 0), "time", np.arange(ntime)
        ).data
        enso[length >= threshold] = phase

    length = get_nan_block_lengths(
        xr.where(enso == fill_value, np.nan, 0, keep_attrs=False),
        "time",
        np.arange(oni.sizes["time"]),
    )
    enso[length >= threshold] = "Neutral"
    return enso[enso.data != fill_value].reindex_like(enso, method="nearest")


def make_enso_mask(nino34=None):
    from xarray.core.missing import _get_nan_block_lengths

    if nino34 is None:
        warnings.warn("Pass ONI directly please", DeprecationWarning)
        nino34 = process_nino34()

    ssta = nino34 - nino34.mean()  # .rolling(time=6, center=True).mean()

    enso = xr.full_like(ssta, fill_value="Neutral", dtype="U8")
    index = ssta.indexes["time"] - ssta.indexes["time"][0]
    en_mask = _get_nan_block_lengths(
        xr.where(ssta > 0.4, np.nan, 0, keep_attrs=False), dim="time", index=index
    ) >= pd.Timedelta("169d")

    ln_mask = _get_nan_block_lengths(
        xr.where(ssta < -0.4, np.nan, 0, keep_attrs=False), dim="time", index=index
    ) >= pd.Timedelta("169d")
    # neut_mask = _get_nan_block_lengths(xr.where((ssta < 0.5) & (ssta > -0.5), np.nan, 0), dim="time", index=index) >= pd.Timedelta("120d")

    enso.loc[en_mask] = "El-Nino"
    enso.loc[ln_mask] = "La-Nina"
    # enso.loc[neut_mask] = "Neutral"

    enso.name = "enso_phase"
    enso.attrs[
        "description"
    ] = "ENSO phase; El-Nino = NINO34 SSTA > 0.4 for at least 6 months; La-Nina = NINO34 SSTA < -0.4 for at least 6 months"

    return enso


def make_enso_transition_mask(oni=None):
    if oni is None:
        warnings.warn("Pass ONI directly please", DeprecationWarning)
        oni = process_oni()

    return mixpods.make_enso_transition_mask(oni)
