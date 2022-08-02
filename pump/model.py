import glob
import time

import cf_xarray as cfxr  # noqa
import dask
import dask.delayed
import dcpy.plots
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import xgcm
import xmitgcm

from . import obs
from .calc import (
    calc_reduced_shear,
    get_dcl_base_Ri,
    get_dcl_base_shear,
    get_euc_max,
    get_mld,
    get_tiw_phase,
    get_tiw_phase_sst,
    tiw_avg_filter_sst,
)
from .constants import section_lons
from .mdjwf import dens
from .plot import plot_depths


def read_mitgcm_coords(dirname):
    import xmitgcm

    h = dict()
    for ff in ["XG", "XC", "YG", "YC", "RC", "RF"]:
        try:
            data = (
                xmitgcm.utils.read_mds(dirname + ff)[ff]
                .copy()
                .squeeze()
                .astype("float")
                .compute()
            )
            attrs = {}
            if "X" in ff:
                h[ff] = (ff, data[0, 40:-40] - 170 - 0.025)
                attrs["axis"] = "X"
            elif "Y" in ff:
                h[ff] = (ff, data[40:-40, 0])
                attrs["axis"] = "Y"
            elif "R" in ff:
                h[ff] = (ff, data[:136])
                attrs["axis"] = "Z"

            if "G" in ff or "F" in ff:
                attrs.update({"c_grid_axis_shift": -0.5})

            h[ff] += (attrs,)

        except (FileNotFoundError):
            print(f"metrics files not available. {dirname + ff}")

    return xr.Dataset(h)


def rename_mitgcm_budget_terms(hb, coords):

    hb_ren = xr.Dataset()
    for var in hb:
        if "x_TH" in var:
            hb_ren[var] = (
                hb[var]
                .drop(hb[var].dims)
                .rename({"longitude": "XG", "latitude": "YC", "depth": "RC"})
            )
        elif "y_TH" in var:
            hb_ren[var] = (
                hb[var]
                .drop(hb[var].dims)
                .rename({"longitude": "XC", "latitude": "YG", "depth": "RC"})
            )
        elif "r_TH" in var:
            hb_ren[var] = (
                hb[var]
                .drop(hb[var].dims)
                .rename({"longitude": "XC", "latitude": "YC", "depth": "RF"})
            )
        elif var in ["DFrI_TH", "KPPg_TH"]:
            hb_ren[var] = (
                hb[var]
                .drop(hb[var].dims)
                .rename({"longitude": "XC", "latitude": "YC", "depth": "RF"})
            )
        elif "TOTTTEND" in var:
            hb_ren[var] = (
                hb[var]
                .drop(hb[var].dims)
                .rename({"longitude": "XC", "latitude": "YC", "depth": "RC"})
            )

        elif "UTEND" in var or "Um" in var or "AB_gU" in var:
            hb_ren[var] = (
                hb[var]
                .drop(hb[var].dims)
                .rename({"longitude": "XG", "latitude": "YC", "depth": "RC"})
            )

        elif "VTEND" in var or "Vm" in var or "AB_gV" in var:
            hb_ren[var] = (
                hb[var]
                .drop(hb[var].dims)
                .rename({"longitude": "XC", "latitude": "YG", "depth": "RC"})
            )

        if "VIS" in var:
            hb_ren[var] = hb_ren[var].rename({"RC": "RF"})

    if "WTHMASS" in hb:
        hb_ren["WTHMASS"] = (
            hb.WTHMASS.drop(["latitude", "longitude"])
            .isel(depth=0)
            .rename({"longitude": "XC", "latitude": "YC", "depth": "RC"})
            .expand_dims("RC")
            .reindex(RC=coords.coords["RC"], fill_value=0)
        )

    hb_ren["time"] = hb.time

    return hb_ren.update(coords.coords)


def mitgcm_sw_prof(depth):
    """MITgcm Shortwave radiation penetration profile."""
    return 0.62 * np.exp(depth / 0.6) + (1 - 0.62) * np.exp(depth / 20)


def make_cartesian(ds):

    renamer = {
        dim: dim.replace("xi", "lon").replace("eta", "lat")
        for dim in ds.dims
        if "xi" in dim or "eta" in dim
    }

    if not renamer:
        return ds

    ds = ds.copy(deep=True)

    # deal with xroms renaming
    synonym = {"xi_v": "xi_rho", "eta_u": "eta_rho"}

    for dim in ds.dims:
        if "xi" not in dim and "eta" not in dim:
            continue

        newdim = renamer[dim]
        if "xi" in dim:
            other = "eta_" + dim.split("_")[1]
        else:
            other = "xi_" + dim.split("_")[1]

        if other not in ds.dims:
            other = synonym[other]

        ds[newdim] = ds[newdim].isel({other: 0}).compute()
    ds = ds.swap_dims(renamer)

    todrop = [var for var in ds.coords if "eta" in var or "xi" in var]
    ds = ds.drop_vars(todrop)

    for dim in ds.dims:
        if "lat" in dim:
            ds[dim].attrs["axis"] = "Y"
        elif "lon" in dim:
            ds[dim].attrs["axis"] = "X"

    return ds


def read_roms_dataset(fnames, **chunk_kwargs):
    chunks = {}

    for sub in ["rho", "u", "v", "psi"]:
        for k, v in chunk_kwargs.items():
            if k == "s" or k == "ocean_time":
                continue
            chunks[f"{k}_{sub}"] = v

    for sub in ["rho", "w"]:
        for k, v in chunk_kwargs.items():
            if k != "s":
                continue
            chunks[f"{k}_{sub}"] = v

    chunks["ocean_time"] = chunk_kwargs["ocean_time"]

    ds = xr.open_mfdataset(
        fnames,
        chunks=chunks,
        parallel=True,
        compat="override",
        data_vars="minimal",
        coords="minimal",
        decode_times=False,
    )

    ds = ds.set_coords(
        [
            var
            for var in ds
            if (
                not ds[var].shape
                or "obc_" in var
                or "mask_" in var
                or "Cs_" in var
                or "Ltracer" in var
                or var in ["pm", "pn", "h", "f", "angle"]
                or (ds[var].ndim == 1 and ds[var].dims[0] == "tracer")
            )
        ]
    )

    # fix time
    ds.ocean_time.attrs["units"] = "seconds since 1957-01-01 00:00"
    ds = xr.decode_cf(ds)
    ds["ocean_time"] = ds.indexes["ocean_time"].to_datetimeindex()

    # make cartesian for easy indexing
    # ds = make_cartesian(ds).cf.guess_coord_axis()
    # ds.cf.decode_vertical_coords()

    return ds.cf.guess_coord_axis()


def read_stations_20(dirname="~/pump/TPOS_MITgcm_fix3/", globstr="*", dayglobstr="0*"):
    metrics = read_metrics(dirname)
    metrics["longitude"] = metrics.longitude - 169.025

    # stationdirname = f"{dirname}/STATION_DATA/Day_{dayglobstr}"

    #    xr.open_mfdataset(
    #        f"{stationdirname}/{globstr}.nc",
    #        parallel=True,
    #        combine="by_coords",
    #        decode_times=False,
    #    )
    #    .sortby("latitude")
    #    .squeeze()
    # )

    station = xr.open_mfdataset(
        f"{dirname}/STATION_DATA/*.zarr",
        parallel=True,
        engine="zarr",
        backend_kwargs={"consolidated": True},
        chunks={"longitude": 3, "latitude": 3},
    ).cf.guess_coord_axis()

    # TODO: for some reason there are duplicated timestamps near the end
    # if ~station.indexes["time"].is_unique:
    #    deduped = station.time.copy(data=~station.indexes["time"].duplicated())
    #    station = station.where(deduped, drop=True)

    # station.time.attrs["long_name"] = ""
    # station.time.attrs["units"] = "seconds since 1999-01-01 00:00"
    # station = xr.decode_cf(station)
    # station["time"] = station.time - pd.Timedelta("7h")

    metrics = metrics.reindex(
        longitude=station.longitude.values,
        latitude=station.latitude.values,
        method="nearest",
    )

    station["KPPg_TH"] = station.KPPg_TH.fillna(0)

    station["Jq_shear"] = 1035 * 3994 * (station.DFrI_TH.fillna(0)) / metrics.RAC
    station["nonlocal_flux"] = 1035 * 3994 * (station.KPPg_TH) / metrics.RAC
    station["dens"] = dens(station.salt, station.theta, np.array([0.0]))
    station["densT"] = dens(np.array([35.0]), station.theta, np.array([0.0]))

    station["mld"] = get_mld(station.dens)
    # station["Jq"] += station.nonlocal_flux.fillna(0)  # add the non-local term too
    station["Jq"] = station.Jq_shear + station.nonlocal_flux

    # station["eucmax"] = get_euc_max(station.u)
    # station.coords["zeuc"] = station.depth - station.eucmax
    # station.zeuc.attrs["long_name"] = "$z - z_{EUC}$"
    # station.zeuc.attrs["units"] = "m"

    station["dJdz"] = station.Jq.differentiate("depth")
    station["dTdt"] = -station.dJdz / 1035 / 3995 * 86400 * 30
    station.dTdt.attrs["long_name"] = "$∂T/∂t = -1/(ρ_0c_p) ∂J_q/∂z$"
    station.dTdt.attrs["units"] = "°C/month"

    station.u.attrs["standard_name"] = "sea_water_x_velocity"
    station.u.attrs["units"] = "m/s"

    station.v.attrs["standard_name"] = "sea_water_y_velocity"
    station.v.attrs["units"] = "m/s"

    station.depth.attrs["unit"] = "m"
    station.depth.attrs["positive"] = "up"

    return station


def read_metrics(dirname):
    """
    If size of the metrics variables are not the same as (longitude, latitude),
    the code assumes that a boundary region has been cut out at the low-end and
    high-end of the appropriate axis.

    If the size in depth-axis is different, then it assumes that the provided depth
    is a slice from surface to the Nth-point where N=len(depth).
    """
    import xmitgcm

    h = dict()
    for ff in ["hFacC", "RAC", "RF", "DXC", "DYC", "XC", "YC"]:
        try:
            h[ff] = (
                xmitgcm.utils.read_mds(dirname + ff)[ff]
                .copy()
                .squeeze()
                .astype("float")
            )
        except (FileNotFoundError):
            print(f"metrics files not available. {dirname + ff}")
            metrics = None
            return xr.Dataset()

    hFacC = h["hFacC"]
    RAC = h["RAC"]
    RF = h["RF"]
    DXC = h["DXC"]
    DYC = h["DYC"]
    longitude = h["XC"][0, :].compute() - 170 - 0.025
    latitude = h["YC"][:, 0].compute()

    del h

    if len(longitude) != RAC.shape[1]:
        dlon = RAC.shape[1] - len(longitude)
        lons = slice(dlon // 2, -dlon // 2)
    else:
        lons = slice(None, None)

    if len(latitude) != RAC.shape[0]:
        dlat = RAC.shape[0] - len(latitude)
        lats = slice(dlat // 2, -dlat // 2)
    else:
        lats = slice(None, None)

    RAC = xr.DataArray(
        RAC[lats, lons],
        dims=["latitude", "longitude"],
        coords={"longitude": longitude, "latitude": latitude},
        name="RAC",
    )
    DXC = xr.DataArray(
        DXC[lats, lons],
        dims=["latitude", "longitude"],
        coords={"longitude": longitude, "latitude": latitude},
        name="DXC",
    )
    DYC = xr.DataArray(
        DYC[lats, lons],
        dims=["latitude", "longitude"],
        coords={"longitude": longitude, "latitude": latitude},
        name="DYC",
    )

    depth = xr.DataArray(
        (RF[1:] + RF[:-1]) / 2,
        dims=["depth"],
        name="depth",
        attrs={"long_name": "depth", "units": "m", "positive": "up", "axis": "Z"},
    )

    dRF = xr.DataArray(
        np.diff(RF.squeeze()),
        dims=["depth"],
        coords={"depth": depth},
        name="dRF",
        attrs={
            "standard_name": "cell_thickness",
            "long_name": "cell_height",
            "units": "m",
        },
    )

    RF = xr.DataArray(
        RF.squeeze(),
        dims=["RF"],
        name="RF",
        attrs={"long_name": "depth", "units": "m", "positive": "up", "axis": "Z"},
    )

    hFacC = xr.DataArray(
        hFacC[:, lats, lons],
        dims=["depth", "latitude", "longitude"],
        coords={
            "depth": depth,
            "latitude": latitude,
            "longitude": longitude,
        },
        name="hFacC",
    )

    metrics = xr.merge([dRF, hFacC, RAC, DXC, DYC])

    metrics["cellvol"] = np.abs(metrics.RAC * metrics.dRF * metrics.hFacC)

    metrics["cellvol"] = metrics.cellvol.where(metrics.cellvol > 0)

    metrics.coords["RF"] = RF

    metrics["rAw"] = xr.DataArray(
        xmitgcm.utils.read_mds(dirname + "/RAW")["RAW"][lats, lons].astype("float32"),
        dims=["latitude", "longitude"],
    )
    metrics["hFacW"] = xr.DataArray(
        xmitgcm.utils.read_mds(dirname + "/hFacW")["hFacW"][:, lats, lons].astype(
            "float32"
        ),
        dims=["depth", "latitude", "longitude"],
    )
    metrics["hFacW"] = metrics.hFacW.where(metrics.hFacW > 0)

    metrics["hFacS"] = xr.DataArray(
        xmitgcm.utils.read_mds(dirname + "/hFacS")["hFacS"][:, lats, lons].astype(
            "float32"
        ),
        dims=["depth", "latitude", "longitude"],
    )

    metrics["drF"] = xr.DataArray(
        xmitgcm.utils.read_mds(dirname + "/DRF")["DRF"].squeeze().astype("float32"),
        dims=["depth"],
    )

    metrics["drC"] = xr.DataArray(
        xmitgcm.utils.read_mds(dirname + "/DRC")["DRC"].squeeze().astype("float32"),
        dims=["RF"],
    )

    metrics["rAs"] = xr.DataArray(
        xmitgcm.utils.read_mds(dirname + "/RAS")["RAS"].squeeze().astype("float32"),
        dims=["latitude", "longitude"],
    )

    metrics["DXG"] = xr.DataArray(
        xmitgcm.utils.read_mds(dirname + "/DXG")["DXG"].squeeze().astype("float32"),
        dims=["latitude", "longitude"],
    )

    metrics["DYG"] = xr.DataArray(
        xmitgcm.utils.read_mds(dirname + "/DYG")["DYG"].squeeze().astype("float32"),
        dims=["latitude", "longitude"],
    )

    # metrics = metrics.isel(depth=slice(budget.sizes["depth"]), depth_left=slice(budget.sizes["depth"]+1))

    return metrics


def rename_metrics(metrics):
    metrics = (
        metrics.drop(["latitude", "longitude", "depth", "dRF"]).rename(
            {"depth_left": "RF", "depth": "RC", "latitude": "YC", "longitude": "XC"}
        )
        # .update(coords.coords)
    )
    metrics["rAw"] = metrics.rAw.rename({"XC": "XG"})
    metrics["rAs"] = metrics.rAs.rename({"YC": "YG"})

    metrics["hFacW"] = metrics.hFacW.rename({"XC": "XG"})
    metrics["hFacS"] = metrics.hFacS.rename({"YC": "YG"})

    metrics["DXG"] = metrics.DXG.rename({"YC": "YG"})
    metrics["DYG"] = metrics.DYG.rename({"XC": "XG"})

    metrics["DXC"] = metrics.DXC.rename({"XC": "XG"})
    metrics["DYC"] = metrics.DYC.rename({"YC": "YG"})

    return metrics


def read_mitgcm_20_year(
    start, stop, surf=False, mombudget=False, heatbudget=False, state=True, chunks=None
):
    if chunks is None:
        chunks = {"latitude": -1, "longitude": 500}
    gcmdir = "/glade/campaign/cgd/oce/people/bachman/TPOS_1_20_20_year/OUTPUT/"  # MITgcm output directory

    # start date for les; noon is a good time (it is before sunrise)
    # sim_time = pd.Timestamp(start)  # pd.Timestamp("2003-01-01 12:00:00")

    # ADD a 5 day buffer here (:] all kinds of bugs at the beginning and end)
    # les_time_length = 366 * 15  # (days); length of time for forcing/pushing files

    # don't change anything here
    output_start_time = pd.Timestamp("1999-01-01")  # don't change
    firstfilenum = (
        (pd.Timestamp(start) - output_start_time)  # add one day offset to avoid bugs
        .to_numpy()
        .astype("timedelta64[D]")
        .astype("int")
    ) + 1
    lastfilenum = (
        (pd.Timestamp(stop) - output_start_time)  # add one day offset to avoid bugs
        .to_numpy()
        .astype("timedelta64[D]")
        .astype("int")
    ) + 1

    def gen_file_list(suffix):
        files = [
            f"{gcmdir}/File_{num:04d}_{suffix}.nc"
            for num in range(firstfilenum, lastfilenum + 1)
        ]
        return files

    print(firstfilenum, lastfilenum)

    files = []
    if surf:
        files += gen_file_list(suffix="etan") + gen_file_list(suffix="surf")
    if mombudget:
        files += gen_file_list(suffix="ub") + gen_file_list(suffix="vb")
    if heatbudget:
        files += gen_file_list(suffix="hb")
    if state:
        files += gen_file_list(suffix="buoy")

    coords = read_mitgcm_coords(gcmdir)

    # grid metrics
    # verify at http://gallery.pangeo.io/repos/xgcm/xgcm-examples/02_mitgcm.html#
    metrics = (
        read_metrics(gcmdir)
        .compute()
        .isel(
            longitude=slice(40, -40),
            latitude=slice(40, -40),
            depth=slice(136),
            depth_left=slice(136),
            RF=slice(136),
        )
        .pipe(rename_metrics)
    )

    ds = (
        xr.open_mfdataset(files, chunks=chunks, combine="by_coords", parallel=True)
        .rename({"latitude": "YC", "longitude": "XC"})
        .update(coords.coords)
    )

    if surf:
        ds["oceQsw"] = ds.oceQsw.fillna(0)
    if mombudget:
        ds["taux"] = 1035 * 2.5 * ds["Um_Ext"].isel(RC=0)
        ds["tauy"] = 1035 * 2.5 * ds["Vm_Ext"].isel(RC=0)

    # if heatbudget:
    #   hb_ren = pump.model.rename_mitgcm_budget_terms(hb, coords).update(coords.coords)

    ds["u"] = ds.u.drop("XC").rename({"XC": "XG"})
    ds["v"] = ds.v.drop("YC").rename({"YC": "YG"})
    ds = ds.drop("depth").rename({"depth": "RC"})

    for var in ["w", "KPP_diffusivity"]:
        ds[var] = ds[var].drop("RC").rename({"RC": "RF"})

    ds["dens"] = dens(ds.salt, ds.theta, np.array([0.0]))

    ds.YG.attrs["axis"] = "Y"
    ds.YC.attrs["axis"] = "Y"
    ds.XG.attrs["axis"] = "X"
    ds.XC.attrs["axis"] = "X"
    ds.RF.attrs["axis"] = "Z"
    ds.RC.attrs["axis"] = "Z"

    grid = xgcm.Grid(
        xr.merge([coords, metrics]),
        periodic=False,
        boundary={"X": "extend", "Y": "extend", "Z": "extend"},
        metrics={
            "X": ("DXC", "DXG"),
            "Y": ("DYG", "DYC"),
            "Z": ("drF", "drC"),
            ("X", "Y"): ("rAw", "rAs"),
        },
    )

    ds["N2"] = 9.81 / 1035 * grid.derivative(ds.dens, "Z")
    ds["S2"] = grid.interp_like(
        grid.derivative(ds.u, "Z") ** 2, ds.N2
    ) + grid.interp_like(grid.derivative(ds.v, "Z") ** 2, ds.N2)
    ds["Ri"] = ds.N2 / ds.S2

    ds = dask.optimize(ds)[0]

    ds["Ri"].data = dask.optimize(ds.Ri)[0].data
    ds["mld"] = get_mld(ds.dens)

    fordcl = ds[["mld", "Ri"]].cf.chunk({"Z": -1}).unify_chunks()
    ds["dcl_base"] = xr.map_blocks(
        get_dcl_base_Ri,
        fordcl,
        template=fordcl.mld,
    )
    ds.dcl_base.attrs["long_name"] = "$z_{Ri}$"

    # ds["dcl_base"] = pump.calc.get_dcl_base_Ri(ds, depth_thresh=-250)
    ds["dcl"] = ds.mld - ds.dcl_base
    ds.dcl.attrs["long_name"] = "DCL"
    return ds, metrics, grid


class Model:
    def __init__(self, dirname, name, kind="mitgcm", full=False, budget=False):
        """
        Create an object that represents one model run.

        Inputs
        ------

        dirname: str
            Location of netCDF output. MITgcm stuff is in /HOLD/
        name: str
            Name for this run.
        kind: str, optional
            mitgcm or ROMS?
        full: bool, optional
            Read in all output files using mfdataset?
        budget: bool, optional
            Read in heat budget terms using mfdataset?
        """

        self.dirname = dirname
        self.kind = kind
        self.name = name

        try:
            if name == "gcm100":
                self.surface = xr.open_mfdataset(
                    f"{dirname}/SURFACE/*.nc",
                    coords="minimal",
                    data_vars="minimal",
                    compat="override",
                    combine="nested",
                    parallel=True,
                    concat_dim="time",
                    chunks={
                        "longitude": 2500,
                        "latitude": 800,
                        "depth": -1,
                        "time": 12,
                    },
                ).squeeze()
            else:
                self.surface = xr.open_dataset(
                    self.dirname + "/obs_subset/surface.nc",
                    chunks={"time": 227, "longitude": 116, "latitude": -1},
                ).squeeze()
        except (FileNotFoundError, OSError):
            self.surface = xr.Dataset()

        try:
            self.annual = xr.open_mfdataset(
                self.dirname + "/obs_subset/annual-mean*.nc", combine="by_coords"
            ).squeeze()
        except (FileNotFoundError, OSError):
            self.annual = xr.Dataset()

        self.domain = dict()
        self.domain["xyt"] = dict()

        if self.domain["xyt"]:
            self.oisst = obs.read_sst(self.domain["xyt"])

        if full:
            self.read_full()
        else:
            self.full = xr.Dataset()
            self.depth = None

        self.update_coords()
        self.read_metrics()

        if budget:
            self.read_budget()
        else:
            self.budget = xr.Dataset()

        self.mean = self.annual  # forgot that I had read this in before!

        class obs_container:
            pass

        self.obs = obs_container()

        self.read_tao()

        try:
            self.johnson = xr.open_dataset(
                dirname + "/obs_subset/johnson-section-mean.nc"
            )
        except FileNotFoundError:
            self.johnson = None

        self.tiw_trange = [
            slice("1995-10-01", "1996-03-01"),
            slice("1996-08-01", "1997-03-01"),
        ]

    def __repr__(self):
        string = f"{self.name} [{self.dirname}]"
        # Add resolution

        return string

    def extract_johnson_sections(self):
        (
            self.full.sel(longitude=section_lons, method="nearest")
            .sel(time=str(self.mid_year))
            .mean("time")
            .load()
            .to_netcdf(self.dirname + "/obs_subset/johnson-section-mean.nc")
        )

    def extract_tao(self):
        region = dict(
            longitude=[-170, -155, -140, -125, -110, -95],
            latitude=[-8, -5, -2, 0, 2, 5, 8],
            method="nearest",
        )
        datasets = [self.full.sel(**region).sel(depth=slice(0, -500)).load()]
        if self.budget:
            print("Merging in budget terms...")
            datasets.append(self.budget.sel(**region).sel(depth=slice(0, -500)).load())
            if not self.full.time.equals(self.budget.time):
                datasets[0] = datasets[0].reindex(time=self.budget.time)

        self.tao = xr.merge(datasets)

        # round lat, lon
        self.tao["latitude"].values = np.array([-8, -5, -2, 0, 2, 5, 8]) * 1.0
        self.tao["longitude"].values = (
            np.array([-170, -155, -140, -125, -110, -95]) * 1.0
        )

        print("Writing to file...")
        (self.tao.load().to_netcdf(self.dirname + "/obs_subset/tao-extract.nc"))

    def read_full(self, preprocess=None):
        start_time = time.time()

        if self.name == "gcm1":
            files = "/Day_*[0-9].nc"
            chunks = {"depth": 50 * 2, "latitude": 69 * 2, "longitude": 215 * 2}
        elif "gcm20" in self.name:
            files = "/File_*buoy.nc"
            # chunks = {"depth": 28 * 5, "latitude": 80 * 5, "longitude": 284 * 5}
            chunks = None
        elif self.name == "gcm100":
            files = "/cmpr_*.nc"
            chunks = {"longitude": 500, "latitude": 160, "depth": -1}
        else:
            chunks = dict(zip(["depth", "latitude", "longitude"], ["auto"] * 3))

        if self.kind == "mitgcm":
            self.full = xr.open_mfdataset(
                self.dirname + files,
                concat_dim="time",
                engine="h5netcdf",
                parallel=True,
                chunks=chunks,
                combine="nested",
                preprocess=preprocess,
            )

            self.full["dens"] = dens(self.full.salt, self.full.theta, np.array([0.0]))

            if self.name == "gcm100":
                self.full["time"] = self.surface.time[::24]

        if self.kind == "roms":
            self.full = xr.Dataset()

        print(
            "Reading all files took {time} seconds".format(
                time=time.time() - start_time
            )
        )

        self.depth = self.full.depth

    def read_metrics(self):
        dirname = self.dirname

        h = dict()
        for ff in ["hFacC", "RAC", "RF"]:
            try:
                h[ff] = xmitgcm.utils.read_mds(dirname + ff)[ff]
            except (FileNotFoundError, OSError):
                print("metrics files not available.")
                self.metrics = None
                return xr.Dataset()

        hFacC = h["hFacC"].copy().squeeze().astype("float32")
        RAC = h["RAC"].copy().squeeze().astype("float32")
        RF = h["RF"].copy().squeeze().astype("float32")

        del h

        RAC = xr.DataArray(
            RAC,
            dims=["latitude", "longitude"],
            coords={"longitude": self.longitude, "latitude": self.latitude},
            name="RAC",
        )

        self.depth = xr.DataArray(
            (RF[1:] + RF[:-1]) / 2,
            dims=["depth"],
            name="depth",
            attrs={"long_name": "depth", "units": "m"},
        )

        dRF = xr.DataArray(
            np.diff(RF.squeeze()),
            dims=["depth"],
            coords={"depth": self.depth},
            name="dRF",
            attrs={"long_name": "cell_height", "units": "m"},
        )

        RF = xr.DataArray(RF.squeeze(), dims=["depth_left"], name="depth_left")

        hFacC = xr.DataArray(
            hFacC,
            dims=["depth", "latitude", "longitude"],
            coords={
                "depth": self.depth,
                "latitude": self.latitude,
                "longitude": self.longitude,
            },
            name="hFacC",
        )

        metrics = xr.merge([dRF, hFacC, RAC])

        metrics["cellvol"] = np.abs(metrics.RAC * metrics.dRF * metrics.hFacC)

        metrics["cellvol"] = metrics.cellvol.where(metrics.cellvol > 0)

        metrics["RF"] = RF

        self.metrics = metrics

    def read_budget(self):

        if self.name == "gcm1":
            chunks = {"depth": 50 * 2, "latitude": 69 * 2, "longitude": 215 * 2}
        else:
            chunks = dict(zip(["depth", "latitude", "longitude"], ["auto"] * 3))

        kwargs = dict(
            engine="h5netcdf", parallel=True, concat_dim="time", combine="nested"
        )

        files = sorted(glob.glob(self.dirname + "Day_*_hb.nc"))
        self.budget = xr.merge(
            [
                xr.open_mfdataset(
                    files,
                    drop_variables=["DFxE_TH", "DFyE_TH", "DFrE_TH"],
                    chunks=chunks,
                    **kwargs,
                ),
                xr.open_mfdataset(self.dirname + "Day_*_sf.nc", **kwargs),
            ]
        )

        self.budget["oceQsw"] = self.budget.oceQsw.fillna(0)

        chunks_dict = dict(zip(self.budget.DFrI_TH.dims, self.budget.DFrI_TH.chunks))
        chunks_dict.pop("time")
        CV = self.metrics.cellvol.chunk(chunks_dict)
        dz = np.abs(self.metrics.dRF[0])
        self.budget["Jq"] = 1035 * 3999 * dz * self.budget.DFrI_TH / CV
        self.budget["Jq"].attrs["long_name"] = "$J_q^t$"
        self.budget["Jq"].attrs["units"] = "W/m$^2$"

    def get_tiw_phase(self, v, debug=False):

        ph = []
        for tt in self.tiw_trange:
            print(tt)
            ph.append(get_tiw_phase(v.sel(time=tt), debug=debug))
            if len(ph) > 1:
                start_num = ph[-2].period.max()
            else:
                start_num = 0
            ph[-1]["period"] += start_num

        phase = xr.merge(ph).drop("variable").reindex(time=v.time)

        return phase.set_coords("period")["tiw_phase"]

    def read_tao(self):
        try:
            self.tao = xr.open_mfdataset(
                self.dirname + "/obs_subset/tao-*extract.nc", combine="by_coords"
            )
        except (FileNotFoundError, OSError):
            self.tao = None
            return

        self.tao["dens"] = dens(self.tao.salt, self.tao.theta, self.tao.depth)
        self.tao = calc_reduced_shear(self.tao)
        self.tao["euc_max"] = get_euc_max(self.tao.u)
        self.tao["mld"] = get_mld(self.tao.dens)
        self.tao["dcl_base_shear"] = get_dcl_base_shear(self.tao)
        self.tao["dcl_base_Ri"] = get_dcl_base_Ri(self.tao)
        self.tao["dens"] = dens(self.tao.salt, self.tao.theta, self.tao.depth)

        if self.metrics:
            CV = self.metrics.cellvol.sel(
                latitude=self.tao.latitude,
                longitude=self.tao.longitude,
                depth=self.tao.depth,
                method="nearest",
            ).assign_coords(**dict(self.tao.isel(time=1).coords))

            dz = np.abs(self.metrics.dRF[0])

            self.tao["Jq"] = 1035 * 3999 * dz * self.tao.DFrI_TH / CV
            self.tao["Jq"].attrs["long_name"] = "$J_q^t$"
            self.tao["Jq"].attrs["units"] = "W/m$^2$"

    def update_coords(self):
        if "latitude" in self.surface:
            ds = self.surface
        elif "latitude" in self.full:
            ds = self.full
        elif "latitude" in self.budget:
            ds = self.budget
        else:
            return None

        self.latitude = ds.latitude
        self.longitude = ds.longitude
        self.time = ds.time
        self.mid_year = np.unique(self.time.dt.year)[1]

        if "depth" in ds.variables and not np.isscalar(ds["depth"]):
            self.depth = ds.depth

        for dim in ["latitude", "longitude", "time"]:
            self.domain["xyt"][dim] = slice(
                getattr(self, dim).values.min(), getattr(self, dim).values.max()
            )

        self.domain["xy"] = {
            "latitude": self.domain["xyt"]["latitude"],
            "longitude": self.domain["xyt"]["longitude"],
        }

    def plot_tiw_summary(self, subset, ax=None, normalize_period=False, **kwargs):

        if ax is None:
            f, axx = plt.subplots(
                9,
                1,
                sharex=True,
                sharey=False,
                constrained_layout=True,
                gridspec_kw=dict(height_ratios=[2] + [1] * 8),
            )
            ax = dict(zip(["sst", "u", "v", "w", "theta", "S2", "N2", "Jq", "Ri"], axx))
            f.set_size_inches((6, 10))

        else:
            axx = list(ax.values())

        cmaps = dict(
            sst=mpl.cm.RdYlBu_r,
            u=mpl.cm.RdBu_r,
            v=mpl.cm.RdBu_r,
            w=mpl.cm.RdBu_r,
            S2=mpl.cm.Reds,
            N2=mpl.cm.Blues,
            Jq=mpl.cm.BuGn_r,
            KT=mpl.cm.Reds,
            Ri=mpl.cm.Reds,
            theta=mpl.cm.RdYlBu_r,
        )

        x = kwargs.get("x")

        handles = dict()
        for aa in ax:
            if aa == "KT":
                pkwargs = dict(norm=mpl.colors.LogNorm())
            elif aa == "S2":
                pkwargs = dict(vmin=0, vmax=5e-4)
            elif aa == "Jq":
                pkwargs = dict(vmax=0, vmin=-300)
            elif aa == "u":
                pkwargs = dict(vmin=-0.8, vmax=0.8)
            elif aa == "v":
                pkwargs = dict(vmin=-0.5, vmax=0.5)
            elif aa == "w":
                pkwargs = dict(vmin=-1.5e-4, vmax=1.5e-4)
            elif aa == "S":
                pkwargs = dict()
            elif aa == "N2":
                pkwargs = dict(vmin=0, vmax=3e-4)
            elif aa == "Ri":
                pkwargs = dict(levels=[0.1, 0.25, 0.35, 0.5])
            else:
                pkwargs = {"robust": True}

            if aa == "sst":
                handles[aa] = subset[aa].plot(
                    ax=ax[aa], y="sst_lat", cmap=cmaps[aa], **kwargs, **pkwargs
                )
            else:
                handles[aa] = (
                    subset[aa]
                    .sel(depth=slice(0, -180))
                    .plot(
                        ax=ax[aa],
                        y="depth",
                        ylim=[-180, 0],
                        cmap=cmaps[aa],
                        **kwargs,
                        **pkwargs,
                    )
                )
                plot_depths(subset, ax=ax[aa], x=x)

        subset.salt.plot.contour(
            ax=ax["theta"], y="depth", levels=12, colors="gray", linewidths=0.5
        )

        for aa in axx[:-1]:
            aa.set_xlabel("")

        for aa in axx[1:]:
            aa.set_title("")

        if x:
            if "phase" in x:
                axx[0].set_xlim([0, 360])

        if normalize_period:
            phase = subset.tiw_phase.copy(deep=True).dropna("time")
            dtdp = (phase.time[-1] - phase.time[0]).astype("float32") / (
                phase[-1] - phase[0]
            )

            phase_times = []
            for pp in [0, 90, 180, 270]:
                tt = subset.time.where(subset.tiw_phase.isin(pp), drop=True).values

                if tt.size == 1:
                    phase_times.append(tt[0])
                else:
                    delta_p = pp - phase[0]
                    delta_t = (dtdp * delta_p).astype("timedelta64[ns]")
                    phase_times.append(phase.time[0].values + delta_t.values)

            if phase[-1] < 359:
                delta_p = 360 - phase[-1]
                delta_t = (dtdp * delta_p).astype("timedelta64[ns]")
                phase_times.append(phase.time[-1].values + delta_t.values)

            assert len(phase_times) >= 4

            dcpy.plots.linex(phase_times, ax=axx, zorder=10, color="k", lw=1)

            # plt.figure()
            # subset.tiw_phase.plot()
            # dcpy.plots.linex(phase_times)
            # dcpy.plots.liney([0, 90, 180, 270, 360])

            axx[0].set_xlim([np.min(phase_times), np.max(phase_times)])

        return handles, ax

    def plot_tiw_composite(
        self, region=dict(latitude=0, longitude=-140), ax=None, ds="tao", **kwargs
    ):

        ds = getattr(self, ds)

        subset = ds.sel(**region)

        tiw_phase = self.get_tiw_phase(subset.v)
        subset = subset.rename({"KPP_diffusivity": "KT"}).where(
            subset.depth < subset.mld - 5
        )

        for vv in ["mld", "dcl_base_shear", "euc_max"]:
            subset[vv] = subset[vv].max("depth")

        phase_bins = np.arange(0, 365, 10)
        grouped = subset.groupby_bins(tiw_phase, bins=phase_bins)
        mean = grouped.mean("time")

        handles, ax = self.plot_tiw_summary(mean, x="tiw_phase_bins")

        ax.get("u").set_xticks([0, 90, 180, 270, 360])

        for _, aa in ax.items():
            aa.grid(True, axis="x")

        return handles, ax

    def plot_dcl(self, region, ds="tao"):

        subset = getattr(self, ds).sel(**region)

        f, axx = plt.subplots(
            6,
            1,
            constrained_layout=True,
            sharex=True,
            gridspec_kw=dict(height_ratios=[1, 1, 5, 5, 5, 5]),
        )

        ax = dict(zip(["v", "Q", "KT", "shear", "N2", "Ri"], axx))

        (
            np.log10(subset.KPP_diffusivity).plot(
                ax=ax["KT"],
                x="time",
                vmin=-6,
                vmax=-2,
                cmap=mpl.cm.GnBu,
                ylim=[-150, 0],
            )
        )

        # dcl_K = (subset.KPP_diffusivity.where(
        #     (subset.depth < (subset.mld - 5))
        #     & (subset.depth > (subset.dcl_base + 5))))
        # dcl_K = dcl_K.where(dcl_K < 1e-2)
        # (dcl_K.mean('depth')
        #  .plot(ax=ax['dcl_KT'], x='time', yscale='log', _labels=False,
        #        label='mean'))
        # (dcl_K.median('depth')
        #  .plot(ax=ax['dcl_KT'], x='time', yscale='log', _labels=False,
        #        ylim=[5e-4, 3e-3], label='median'))

        # ax['dcl_KT'].set_ylabel('DCL $K$')
        # ax['dcl_KT'].legend()

        subset.oceQnet.plot(ax=ax["Q"], x="time", _labels=False)

        subset.v.isel(depth=1).plot(ax=ax["v"], x="time", _labels=False)

        (subset.shear**2).plot(
            ax=ax["shear"],
            x="time",
            ylim=[-150, 0],
            robust=True,
            cmap=mpl.cm.RdYlBu_r,
            norm=mpl.colors.LogNorm(1e-6, 1e-3),
        )
        (subset.N2).plot(
            ax=ax["N2"],
            x="time",
            ylim=[-150, 0],
            robust=True,
            cmap=mpl.cm.RdYlBu_r,
            norm=mpl.colors.LogNorm(1e-6, 1e-3),
        )

        inv_Ri = 1 / (subset.N2 / subset.shear**2)
        inv_Ri.attrs["long_name"] = "Inv. Ri"
        inv_Ri.attrs["units"] = ""

        (inv_Ri).plot(
            ax=ax["Ri"],
            x="time",
            ylim=[-150, 0],
            robust=True,
            cmap=mpl.cm.RdBu_r,
            center=4,
        )
        (inv_Ri).plot.contour(
            ax=ax["Ri"],
            x="time",
            ylim=[-150, 0],
            levels=[4],
            colors="gray",
            linewidths=0.5,
        )

        for axx0 in [ax["KT"], ax["shear"], ax["N2"]]:
            subset.euc_max.plot(ax=axx0, color="k", lw=1, _labels=False)
            subset.dcl_base_shear.plot(ax=axx0, color="gray", lw=1, _labels=False)
            (subset.mld - 5).plot(ax=axx0, color="k", lw=0.5, _labels=False)

        ((subset.mld - 5).plot(ax=ax["Ri"], color="k", lw=0.5, _labels=False))
        (subset.euc_max.plot(ax=ax["Ri"], color="k", lw=0.5, _labels=False))
        ax["v"].set_ylabel("v")
        ax["Q"].set_ylabel("$Q_{net}$")
        axx[0].set_title(ax["KT"].get_title())
        [aa.set_title("") for aa in axx[1:]]
        [aa.set_xlabel("") for aa in axx]

        ax["v"].axhline(0, color="k", zorder=-1, lw=1, ls="--")
        ax["Q"].axhline(0, color="k", zorder=-1, lw=1, ls="--")

        f.set_size_inches((8, 8))
        dcpy.plots.label_subplots(ax.values())

    def summarize_tiw_periods(self, subset):
        if "tiw_phase" not in subset:
            subset = xr.merge([subset, self.get_tiw_phase(subset.v)])
        if "sst" not in subset:
            subset["sst"] = self.surface.theta.sel(
                longitude=subset.longitude.values, method="nearest"
            ).rename({"latitude": "sst_lat"})

        def _plot_func(subset, period):
            _, _, f = self.plot_tiw_summary(
                subset.where(subset.period == period, drop=True)
                .drop("period")
                .assign_coords(period=period),
                x="time",
                normalize_period=True,
            )

            f.savefig(
                f"../images/{self.name}-tiw-period"
                f"-{subset.latitude.values}"
                f"-{np.abs(subset.longitude.values)}"
                f"-{period:02.0f}.png",
                dpi=200,
            )

        periods = subset.period.dropna("time")
        subset = dask.delayed(subset)
        return [
            dask.delayed(_plot_func)(subset, period) for period in np.unique(periods)
        ]

    def get_quantities_for_composite(self, longitudes=[-110, -125, -140, -155]):
        surf = (
            self.surface.sel(longitude=longitudes, method="nearest")
            .assign_coords(longitude=longitudes)
            .drop("depth")
        )
        sst = surf.theta.sel(longitude=longitudes).chunk({"longitude": 1, "time": -1})
        sstfilt = tiw_avg_filter_sst(sst, "bandpass")  # .persist()

        # surf["theta"] = (
        #    sst.pipe(xfilter.lowpass, coord="time", freq=1/7, cycles_per="D", num_discard=0, method="pad")
        # )

        Tx = self.surface.theta.differentiate("longitude")
        Ty = self.surface.theta.differentiate("latitude")
        gradT = (
            np.hypot(Tx, Ty)
            .sel(longitude=longitudes, method="nearest")
            .assign_coords(longitude=longitudes)
            .drop_vars("depth")
        )

        tiw_phase, period, tiw_ptp = dask.compute(
            get_tiw_phase_sst(
                sstfilt.chunk({"longitude": 1}),
                gradT.chunk({"longitude": 1, "time": -1}),
            ),
        )[0]

        return surf, sst, sstfilt, gradT, tiw_phase, period, tiw_ptp
