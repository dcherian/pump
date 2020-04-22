import dcpy

import numpy as np
import pandas as pd
import tqdm
import xarray as xr

from . import mdjwf

from .constants import *

root = "/glade/work/dcherian/pump/"


def read_all(domain=None):
    johnson = read_johnson()
    tao = read_tao(domain)
    sst = read_sst(domain)
    oscar = read_oscar(domain)

    return [johnson, tao, sst, oscar]


def read_johnson(filename=root + "/obs/johnson-eq-pac-adcp.cdf"):
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


def read_tao_adcp(domain=None, freq="dy"):

    if freq == "dy":
        adcp = xr.open_dataset(root + "/obs/tao/adcp_xyzt_dy.cdf").rename(
            {"lon": "longitude", "lat": "latitude", "U_1205": "u", "V_1206": "v"}
        )
    elif freq == "hr":
        afiles = [
            root + "/obs/tao/adcp0n" + lon + "_hr.cdf"
            for lon in ["156e", "165e", "170w", "140w", "110w"]
        ]

        ds = []
        for file in tqdm.tqdm(afiles):
            ds.append(
                xr.open_dataset(file)
                .drop(["QU_5205", "QV_5206"])
                .rename(
                    {
                        "lon": "longitude",
                        "lat": "latitude",
                        "u_1205": "u",
                        "v_1206": "v",
                    }
                )
            )
        adcp = xr.merge(xr.align(*ds, join="outer"))

    adcp = adcp.chunk({"latitude": 1, "longitude": 1})

    for vv in adcp:
        adcp[vv] /= 100
        adcp[vv].attrs["long_name"] = vv
        adcp[vv].attrs["units"] = "m/s"
        adcp[vv] = adcp[vv].where(np.abs(adcp[vv]) < 1000)

    adcp["longitude"] -= 360
    adcp["depth"] *= -1
    adcp["depth"].attrs["units"] = "m"
    adcp["u"].attrs["units"] = "m/s"
    adcp["v"].attrs["units"] = "m/s"

    if domain is not None:
        adcp = adcp.sel(**domain)
    else:
        adcp = adcp.sel(latitude=0)

    return adcp.dropna("longitude", how="all").dropna("depth", how="all")


def tao_read_and_merge(suffix, kind):
    """ read non-ADCP files. """

    if kind == "temp":
        prefix = "t"
        renamer = {"T_20": "T"}

    elif kind == "cur":
        prefix = "cur"
        renamer = {"U_320": "u", "V_321": "v"}

    ds = []
    tfiles = [
        f"{root}/obs/tao/{prefix}0n{lon}_{suffix}.cdf"
        for lon in ["156e", "165e", "170w", "140w", "110w"]
    ]
    for file in tqdm.tqdm(tfiles):
        try:
            ds.append(xr.load_dataset(file)[list(renamer.keys())])
        except FileNotFoundError:
            pass

    merged = xr.merge(xr.align(*ds, join="outer")).rename(
        {"lon": "longitude", "lat": "latitude"}
    )

    merged["longitude"] -= 360
    merged["depth"] *= -1
    return merged.rename_vars(renamer)


def tao_merge_10m_and_hourly(kind):
    """ Merge 10minute and hourly data into one record. """
    m10 = tao_read_and_merge("10m", kind)
    hr = tao_read_and_merge("hr", kind)

    # resample is really slow because it doesn't know about dask.
    # instead reindex to a 10min freq and use rolling.
    new_index = pd.date_range(
        start=m10.time[0].dt.round("H").values,
        end=m10.time[-1].dt.round("H").values,
        freq="10min",
    )
    m10 = m10.reindex(time=new_index)
    m10hr = (
        m10.chunk({"longitude": 1})
        .rolling(time=6, min_periods=4)
        .construct("window_dim", stride=6)
        .mean("window_dim")
    )

    new_hourly_index = pd.date_range(
        start=np.min([m10.time[0].values, hr.time[0].values]),
        end=np.max([m10.time[-1].values, hr.time[-1].values]),
        freq="H",
    )

    concat = xr.concat(
        [m10hr.reindex(time=new_hourly_index), hr.reindex(time=new_hourly_index)],
        dim="concat",
    )

    # # T at 170W is only available at 10min frequency
    # ds.append(xr.load_dataset(root+'/obs/tao/t0n170w_10m.cdf')['T_20']
    #           .resample(time='H').mean('time'))

    # adcp = read_tao_adcp(freq='hr')
    # return xr.merge(ds)
    return concat.mean("concat").squeeze()


def read_eq_tao_cur_hr():
    return tao_merge_10m_and_hourly("cur") / 100


def read_eq_tao_temp_hr():
    """ Read hourly resolution temperature for equatorial moorings. """

    # sfiles = [root+'/obs/tao/s0n'+lon+'_hr.cdf'
    #           for lon in ['156e', '165e', '140w', '110w', '170w']]
    # for file in tqdm.tqdm(sfiles):
    #     ds.append(xr.open_dataset(file)['S_41'])

    return tao_merge_10m_and_hourly("temp").T


def read_tao(domain=None):
    tao = xr.open_mfdataset(
        [
            root + "/obs/tao/" + ff
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
            [root + "/obs/oisst/sst.day.mean." + str(yy) + ".nc" for yy in years],
            parallel=True,
        )
    else:
        sst = xr.open_mfdataset(root + "/obs/oisst/sst.day.mean.*.nc", parallel=True)

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

    oscar = dcpy.oceans.read_oscar(root + "/obs/oscar/").rename(
        {"lat": "latitude", "lon": "longitude"}
    )
    oscar["longitude"] = oscar["longitude"] - 360
    oscar = oscar.sortby("latitude")

    if domain is not None:
        return oscar.sel(**domain)
    else:
        return oscar


def read_argo():

    dirname = root + "/obs/argo/"
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
    nino34 = process_esrl_index("nina34.data")

    nino34.to_netcdf(root + "/obs/nino34.nc")


def process_oni():
    oni = process_esrl_index("oni.data", skipfooter=8)
    return oni
    # oni.to_netcdf(root + "/obs/oni.nc")


def process_esrl_index(file, skipfooter=3):
    """ Read and make xarray version of climate indices from ESRL."""

    month_names = (
        pd.date_range("01-Jan-2001", "31-Dec-2001", freq="MS")
        .to_series()
        .dt.strftime("%b")
        .values.astype(str)
    )

    index = pd.read_csv(
        root + "/obs/" + file,
        index_col=0,
        names=month_names,
        delim_whitespace=True,
        skiprows=1,
        na_filter=False,
        skipfooter=skipfooter,
        dtype=np.float32,
    )

    flat = index.stack().reset_index()
    flat["time"] = pd.date_range(
        "01-jan-" + str(flat["level_0"].iloc[0]),
        "01-Jan-" + str(flat["level_0"].iloc[-1] + 1),
        freq="M",
    )
    da = (
        flat.drop(["level_0", "level_1"], axis=1)
        .rename({0: "index"}, axis=1)
        .set_index("time")
        .to_xarray()
    )

    return da.where(da > -90)["index"]


def read_jra():
    files = f"{root}/make_TPOS_MITgcm/JRA_FORCING/combined/JRA55DO_*[a-z].nc"

    jra = xr.open_mfdataset(files, combine="by_coords", decode_times=False, chunks={"time": 1}).rename(
        {"lat": "latitude", "lon": "longitude"}
    )

    jra["time"] = jra.time - jra.time[0]
    jra.time.attrs["units"] = "days since 1995-09-01"

    jra["longitude"] -= 360

    jra = jra.sel(longitude=slice(-170, -95), latitude=slice(-12, 12))

    return xr.decode_cf(jra)


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


def read_tao_zarr(kind="gridded"):

    if kind not in ["gridded", "merged", "ancillary"]:
        raise ValueError(
            f"'kind' must be one of ['gridded', 'merged']. Received {kind!r}"
        )

    if kind == "merged":
        tao = xr.open_zarr("tao_eq_hr_merged_cur.zarr", consolidated=True)
    elif kind == "gridded":
        tao = xr.open_zarr("tao_eq_hr_gridded.zarr")
    elif kind == "ancillary":
        tao = xr.open_zarr("tao-gridded-ancillary.zarr")

    tao = tao.chunk({"depth": -1, "time": 10000})
    tao["dens"] = dcpy.eos.dens(35, tao.T, tao.depth)
    return tao
