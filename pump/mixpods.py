import ast
import datetime
import glob
import os
import warnings

import cf_xarray as cfxr
import dcpy
import flox
import fsspec
import gsw_xarray
import holoviews as hv
import hvplot.xarray  # noqa
import intake
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datatree import DataTree
from flox.xarray import xarray_reduce

import xarray as xr

from .obs import process_oni

ROOT = "/glade/campaign/cgd/oce/projects/pump/cesm/"

ENSO_COLORS_RGB = {
    "El-Nino warm": (177, 0, 19),
    "El-Nino cool": (255, 108, 50),
    "La-Nina warm": (55, 150, 202),
    "La-Nina cool": (15, 24, 139),
}
ENSO_COLORS = {k: np.array(v) / 255 for k, v in ENSO_COLORS_RGB.items()}
ENSO_TRANSITION_PHASES = ENSO_COLORS.keys()
LOAD_VARNAMES = [
    "sea_water_x_velocity",
    # TODO fix cf-xarray bug, after selection, vars dont remain as coordinates
    # "eucmax",
    # "mldT",
    "n2s2pdf",
    "S2",
    "N2T",
    "ocean_vertical_heat_diffusivity",
    "eps",
    "chi",
    "Jb",
    "eps_ri",
    "eps_n2s2",
    "Rig_T",
]
DEPTH_CHIPODS = [-89.0, -69.0, -59.0, -49.0, -39.0, -29.0]


# TODO: delete
def ddz(data, grid, h):
    Δ = grid.diff(data, "Z")
    Δz = grid.interp(grid.get_metric(data, "Z"), "Z")
    return (Δ / Δz).rename(f"d{data.name}dz")


def assert_z_is_normalized(ds):
    for z in set(ds.cf.axes["Z"]) & set(ds.dims):
        assert ds[z].attrs["positive"] == "up", ds[z].attrs
        assert (ds[z].data < 1).all(), f"{z!r} is not all negative"
        assert ds.indexes[z].is_monotonic_increasing, f"{z} is not monotonic increasing"


def grouped_median(
    group_idx, array, *, axis=-1, size=None, fill_value=None, dtype=None
):
    import numpy_groupies as npg

    return npg.aggregate_numpy.aggregate(
        group_idx,
        array,
        func=np.nanmedian,
        axis=axis,
        size=size,
        fill_value=fill_value,
        dtype=dtype,
    )


agg_median = flox.aggregations.Aggregation(
    name="median",
    numpy=grouped_median,
    fill_value=None,
    chunk=None,
    combine=None,
)


def prepare(ds, grid=None, sst_nino34=None, oni=None):
    """
    Prepare an input dataset for miχpod diagnostics.

    Parameters
    ----------
    ds: xr.Dataset
        Dataset with "densT", "sea_water_x_velocity", "sea_water_y_velocity"
        optionally with "N2T", "S2"
    grid: xgcm.Grid, optional
        Grid object; neccessary if N2 or S2 need to be calculated.
    sst_nino34: xr.DataArray, optional
        Monthly mean SST in NINO3.4 region. Used to estimate ONI
        if "oni" is not provided.
    oni: xr.DataArray, optional
        Monthly Oceanic Nino Index timeseries. Used to estimate
        "enso_transition" labels..

    Returns
    -------
    xr.Dataset
        with "shred2", "N2T", "S2", "Rig_T", "eucmax", "oni", "enso_transition"
        oni and enso_transition are reindexed from monthly frequency to ds.time
    """

    # dens = ds.cf["sea_water_potential_density"]

    out = ds.copy(deep=True)
    dens = out["densT"]  # Since that's what Warner & Moum do.
    u = out.cf["sea_water_x_velocity"]
    v = out.cf["sea_water_y_velocity"]
    T = out.cf["sea_water_potential_temperature"]
    S = out.cf["sea_water_salinity"]

    assert_z_is_normalized(out)

    with xr.set_options(arithmetic_join="exact"):
        # dρdz = ddz(dens, grid, ds.h)
        # dudz = ddz(u, grid, ds.hu)
        # dvdz = ddz(v, grid, ds.hv)
        if "N2" not in out:
            assert grid is not None
            dρdz = grid.derivative(out["dens"], "Z")
            out["N2"] = -9.81 / grid.interp_like(out["dens"], like=dρdz) * dρdz

            out["N2"].attrs["long_name"] = "$N^2$"
            out["N2"].attrs["units"] = "s$^{-2}$"

        if "Tz" not in out:
            assert grid is not None
            out["Tz"] = grid.derivative(T, "Z")
            out["Tz"].attrs["long_name"] = "$T_z$"
            out["Tz"].attrs["units"] = "Cm$^{-1}$"

        if "Sz" not in out:
            assert grid is not None
            out["Sz"] = grid.derivative(S, "Z")
            out["Sz"].attrs["long_name"] = "$S_z$"
            out["Sz"].attrs["units"] = "m$^{-1}$"

        if "N2T" not in out:
            assert grid is not None
            dρdz = grid.derivative(dens, "Z")
            out["N2T"] = -9.81 / grid.interp_like(dens, like=dρdz) * dρdz

            out["N2T"].attrs["long_name"] = "$N_T^2$"
            out["N2T"].attrs["units"] = "s$^{-2}$"

        if "S2" not in out:
            assert grid is not None
            dudz = grid.derivative(u, "Z")
            dvdz = grid.derivative(v, "Z")

            dvdz = xgcm_interp_to(grid, dvdz, axis="Y", to="center")

            out["S2"] = dudz**2 + dvdz**2
            out["S2"].attrs["long_name"] = "$S^2$"
            out["S2"].attrs["units"] = "s$^{-2}$"

        out["shred2"] = out.S2 - 4 * out.N2T
        out.shred2.attrs["long_name"] = "$Sh_{red}^2$"
        out.shred2.attrs["units"] = "$s^{-2}$"

        out["Rig_T"] = out.N2T / out.S2
        out.Rig_T.attrs["long_name"] = "$Ri^g_T$"
        out.Rig_T.attrs.pop("units", None)

        out["Rig"] = out.N2 / out.S2
        out.Rig.attrs["long_name"] = "$Ri^g$"
        out.Rig.attrs.pop("units", None)

    out["tau"] = np.hypot(
        out.cf["surface_downward_x_stress"], out.cf["surface_downward_y_stress"]
    )
    del out["tau"].attrs["standard_name"]

    ueq = u.cf.sel(Z=slice(-350, -10))
    if "Y" in ueq.cf.axes:
        ueq = ueq.cf.sel(Y=0, method="nearest", drop=True)
    out.coords["eucmax"] = euc_max(ueq)

    out.coords["mldT"] = get_mld_tao_theta(
        out.cf["sea_water_potential_temperature"].reset_coords(drop=True)
    )

    Z = out.S2.reset_coords().cf["Z"]
    out.coords["dcl_mask"] = (Z > (out.eucmax + 5)) & (Z < (out.mldT - 5))
    out.dcl_mask.attrs = {"description": "True when 5m below mldT and above eucmax."}

    assert out.dcl_mask.ndim == 2
    assert out.eucmax.dims == ("time",)
    assert out.mldT.dims == ("time",)

    if sst_nino34 is not None and oni is not None:
        raise ValueError("Provide one of 'sst_nino34' or 'oni'.")
    if sst_nino34 is not None and oni is None:
        oni = calc_oni(sst_nino34)
    if oni is not None:
        oni = oni.rename("oni")
        enso_transition = make_enso_transition_mask(oni).rename("enso_transition")
        out.coords.update(
            xr.merge([oni, enso_transition]).reindex(time=ds.time, method="nearest")
        )

    if "eps" not in out:
        assert grid is not None
        add_turbulence_quantities(out, grid)

    if "sea_surface_temperature" not in ds.cf:
        out["sst"] = (
            out.cf["sea_water_potential_temperature"]
            .cf.sel(Z=0, method="nearest")
            .reset_coords(drop=True)
        )
        out["sst"].attrs.update(
            {"long_name": "$SST$", "standard_name": "sea_surface_temperature"}
        )

    return out


def euc_max(u):
    euc_max = u.cf.idxmax("Z")
    euc_max.attrs.clear()
    euc_max.attrs["units"] = "m"
    euc_max.attrs["long_name"] = "EUC maximum"
    if "positive" in u.cf["Z"].attrs:
        euc_max.attrs["positive"] = u.cf["Z"].attrs["positive"]
    return euc_max


def find_pdf_contour(density, targets):
    """
    Finds PDF contour that encloses target fraction of the samples.
    """

    from collections.abc import Iterable

    from scipy.optimize import minimize_scalar

    def cost(x, density, areas, target):
        within = np.sum(np.where(density.data > x, density.data * areas.data, 0))
        # print(x, within, (within - target)**2, target)
        if within < 0:
            within = -np.inf
        if within >= 0.99:
            within = np.inf
        return (within - target) ** 2

    if not isinstance(targets, Iterable):
        targets = (targets,)

    return tuple(
        minimize_scalar(
            cost,
            args=(density, density.bin_areas, target),
            tol=0.001,
            method="golden",
        ).x
        for target in targets
    )


def to_density(counts, dims=("N2T_bins", "S2_bins")):
    total = counts.sum(dims)
    area = 1
    for dim in dims:
        area = area * counts[dim].copy(
            data=np.array([i.right - i.left for i in counts[dim].data])
        )
    density = counts / total / area
    density.coords["bin_areas"] = area
    return density


def reindex_Z_to(us, other):
    (Zname,) = us.cf.axes["Z"]
    if Zname != other.name:
        # avoid a warning about indexes
        us = us.cf.rename(Z=other.name)
    return us.cf.reindex(Z=other, method="nearest")


def pdf_N2S2(data, coord_is_center=False):
    """
    Calculate pdf of data in S2-4N2 space.
    """

    data = data.copy().reset_coords()
    bins = np.arange(-5, -2.01, 0.1)
    index = pd.IntervalIndex.from_breaks(bins, closed="left")

    assert_z_is_normalized(data)
    do_enso_phase = "enso_transition" in data.variables

    data["S2"] = data.S2.where(data.dcl_mask)

    by = [np.log10(4 * data.N2T), np.log10(data.S2)]
    expected_groups = [index, index]
    isbin = [True, True]

    if "enso_transition" in data.variables:
        by.append(data.enso_transition)
        expected_groups.append(None)
        isbin.append(False)

    # TODO: assert specific dimensions instead
    assert data.S2.ndim < 3
    assert data.N2T.ndim < 3

    enso_kwargs = dict(expected_groups=tuple(expected_groups), isbin=isbin)

    enso_counts = xarray_reduce(data.S2, *by, func="count", **enso_kwargs)

    to_concat = []
    if do_enso_phase:
        counts_all = enso_counts.sum("enso_transition", keepdims=True).assign_coords(
            enso_transition=["none"]
        )
        to_concat.append(counts_all)

        to_concat.append(enso_counts)

        all_enso = (
            enso_counts.sel(
                enso_transition=[
                    "El-Nino cool",
                    "El-Nino warm",
                    "La-Nina cool",
                    "La-Nina warm",
                ]
            )
            .sum("enso_transition", keepdims=True)
            .assign_coords(enso_transition=["all"])
        )
        to_concat.append(all_enso)
    else:
        to_concat.append(enso_counts.expand_dims(enso_transition=["none"]))

    counts = xr.concat(to_concat, dim="enso_transition").rename(
        {"enso_transition": "enso_transition_phase"}
    )
    counts.N2T_bins.attrs["long_name"] = "log$_{10} 4N_T^2$"
    counts.S2_bins.attrs["long_name"] = "log$_{10} S^2$"

    if do_enso_phase:
        counts["enso_transition_phase"].attrs = {
            "description": (
                "El-Nino transition phase as defined by Warner and Moum (2019)."
                " 'none' uses all available data."
                " 'all' sums over all ENSO transition phases"
            )
        }

    counts.attrs = {"long_name": "$P(S^2, 4N_T^2)$"}
    with xr.set_options(keep_attrs=True):
        density = to_density(counts, dims=("N2T_bins", "S2_bins"))

    if coord_is_center:
        density["N2T_bins"] = (bins[1:] + bins[:-1]) / 2
        density["S2_bins"] = (bins[1:] + bins[:-1]) / 2

    out = xr.Dataset(data_vars={"n2s2pdf": density})

    if "eps" in data:
        epsZ = data.eps.cf["Z"].cf.sel(Z=slice(-69, -29))
        # χpod data are available at a subset of depths
        newby = [
            np.log10(4 * reindex_Z_to(data["N2"], epsZ)),
            np.log10(reindex_Z_to(data["S2"], epsZ)),
        ]
        newby.extend(by[2:])
        eps = data.eps.cf.sel(Z=epsZ)

        if "enso_transition" in data.variables:
            out["eps_n2s2"] = xr.concat(
                [
                    xarray_reduce(eps, *newby, func=func, **enso_kwargs)
                    .rename({"enso_transition": "enso_transition_phase"})
                    .expand_dims(stat=[func])
                    for func in ["mean", "count"]
                ],
                dim="stat",
            )

        Ri = reindex_Z_to(data.Rig_T, epsZ)
        # Ri_bins = np.exp(np.logspace(np.log10(0.025), np.log10(2), 11))
        Ri_bins = 10 ** np.arange(-1.6, 0.4, 0.2)  # in log space from Sally
        # Ri_bins = np.array([0.02, 0.04, 0.06, 0.1, 0.3, 0.5, 0.7, 1.5, 2])
        Ri_kwargs = {
            "expected_groups": (Ri_bins, None),
            "isbin": (True, False),
        }

        eps_ri_concat = []
        if do_enso_phase:
            eps_ri = xr.concat(
                [
                    xarray_reduce(
                        eps, Ri, data.enso_transition, func=func, **Ri_kwargs
                    ).expand_dims(stat=[func])
                    for func in ["mean", "std", "count"]
                ],
                dim="stat",
            ).rename({"enso_transition": "enso_transition_phase"})
            eps_ri_concat.append(eps_ri)

        eps_ri_noen = xr.concat(
            [
                xarray_reduce(
                    eps, Ri, expected_groups=Ri_bins, isbin=True, func=func
                ).expand_dims(stat=[func])
                for func in ["mean", "std", "count"]
            ],
            dim="stat",
        ).assign_coords({"enso_transition_phase": "none"})

        out["eps_ri"] = xr.concat(
            [*eps_ri_concat, eps_ri_noen], dim="enso_transition_phase"
        )

    return out


def daily_composites(ds):
    to_composite = (
        ds.cf[
            [
                "eps",
                "ocean_vertical_heat_diffusivity",
                "chi",
                "Jb",
                "Jq",
            ]
        ]
        .cf.rename({"ocean_vertical_heat_diffusivity": "KT"})
        .cf.reindex(Z=DEPTH_CHIPODS, method="nearest")
        .cf.rename({"Z": "depth"})
        .load()
    )

    others = (
        ds.cf[["S2", "N2", "Rig", "Rig_T", "tau"]]
        .cf.reindex(Z=DEPTH_CHIPODS, method="nearest")
        .load()
    )

    if "depth" not in others.dims:
        others = others.cf.rename(Z="depth")

    daily_composites = xarray_reduce(
        xr.merge([to_composite, others], join="exact"),
        ds.time.dt.hour,
        ds.tau.load(),
        expected_groups=(None, pd.IntervalIndex.from_breaks([0, 0.04, 0.075, np.inf])),
        func=agg_median,
    )

    return daily_composites


def plot_n2s2pdf(da, targets=(0.5, 0.75), pcolor=True, **kwargs):
    da = da.squeeze()
    levels = find_pdf_contour(da, targets=targets)
    if pcolor:
        da.plot(robust=True, **kwargs)
    cs = da.reset_coords(drop=True).plot.contour(levels=levels, **kwargs)
    dcpy.plots.line45(ax=kwargs.get("ax", None), ls="--", lw=1.25)
    return cs


def hvplot_n2s2pdf(da, targets=(0.5, 0.75), pcolor=True, **kwargs):
    da = da.squeeze().copy()
    da["N2T_bins"] = pd.IntervalIndex(da.N2T_bins.data).mid.to_numpy()
    da["S2_bins"] = pd.IntervalIndex(da.S2_bins.data).mid.to_numpy()
    levels = find_pdf_contour(da, targets=targets)
    if pcolor:
        da.hvplot.quadmesh(robust=True, **kwargs)
    cs = da.reset_coords(drop=True).hvplot.contour(
        levels=levels, colorbar=pcolor, muted_alpha=0, **kwargs
    )

    return cs * hv.Slope(1, 0)


def plot_stability_diagram(
    ds,
    hue="enso_transition_phase",
    targets=(0.5,),
    title=None,
    ax=None,
    add_legend=True,
    **kwargs,
):
    if ax is None:
        ax = plt.gca()

    handles = []
    labels = []

    assert hue in ds.dims

    # Note this groupby sorts the hue labels...
    for label, group in ds["n2s2pdf"].groupby(hue):
        group["N2T_bins"] = group["N2T_bins"].copy(
            data=pd.IntervalIndex(group["N2T_bins"].data).mid
        )
        group["S2_bins"] = group["S2_bins"].copy(data=group["N2T_bins"].data)

        default_color = (
            ENSO_COLORS[label]
            if hue == "enso_transition_phase"
            else next(ax._get_lines.prop_cycler)["color"]
        )
        color = kwargs.pop("color", (default_color,))

        cs = plot_n2s2pdf(
            group.squeeze(),
            targets=targets,
            colors=color,
            pcolor=False,
            **kwargs,
            ax=ax,
        )
        handles.append(cs.legend_elements()[0][0])
        labels.append(label)

    if add_legend:
        ax.legend(handles, labels, bbox_to_anchor=(1, 1))
    ax.set_xlim([-4.5, -2.5])
    ax.set_ylim([-4.5, -2.5])
    if title:
        ax.set_title(title)


def plot_stability_diagram_by_dataset(datasets, fig=None):
    if fig is None:
        fig = plt.figure(constrained_layout=True, figsize=(8, 4))

    ax = fig.subplots(1, len(datasets), sharex=True, sharey=True)
    for (name, sim), (iaxis, axis) in zip(datasets.items(), enumerate(ax)):
        plot_stability_diagram(
            sim.drop_sel(enso_transition_phase=["all", "none", "____________"]),
            linewidths=2,
            ax=axis,
            title=name,
            add_legend=(iaxis == len(datasets) - 1),
        )
        dcpy.plots.clean_axes(ax)


def plot_stability_diagram_by_phase(datasets, obs="TAO", fig=None):
    if fig is None:
        fig = plt.figure(figsize=(12, 4), constrained_layout=True)

    axx = fig.subplots(1, 4, sharex=True, sharey=True)

    ax = dict(zip(ENSO_TRANSITION_PHASES, axx))

    merged = (
        xr.Dataset(
            {
                k: ds["n2s2pdf"].cf.drop_vars(
                    ["latitude", "longitude"], errors="ignore"
                )
                for k, ds in datasets.items()
            }
        )
        .to_array("dataset")
        .to_dataset(name="n2s2pdf")
    )
    last_phase = list(ENSO_TRANSITION_PHASES)[-1]
    for phase in ENSO_TRANSITION_PHASES:
        plot_stability_diagram(
            merged.sel(enso_transition_phase=phase),
            hue="dataset",
            ax=ax[phase],
            add_legend=(phase == last_phase),
            linewidths=2,
        )
        ax[phase].text(
            x=0.35, y=0.03, s=phase.upper() + "ING", transform=ax[phase].transAxes
        )

    dcpy.plots.clean_axes(axx)


def calc_oni(monthly):
    # https://origin.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php

    # Calculate a monthly 30-year climatological value,
    # updated every 5 years (1856-1885, ..., 1986-2015, 1991-2020).
    nyears_in_window = 30

    climatology = (
        monthly
        # 30 year rolling windows
        .rolling(time=nyears_in_window * 12, center=True)
        .construct(time="month")
        .assign_coords(month=np.concatenate([np.arange(1, 13)] * nyears_in_window))
        # climatology in those 30 year windows
        .groupby("month")
        .mean()
        # decimate appropriately
        .isel(time=slice(4 * 12, None, 5 * 12))
    )

    # Calculate a monthly anomaly with years centered in the climatology
    # (1871-1875 uses 1856-1885 climo, ..., 1996-2000 uses 1981-2010,
    # 2001-2005 uses 1986-2015 climo, 2006-2010 uses 1991-2020 climo,
    # 2011-2025 also uses 1991-2020 climo because 1996-2025 climo does not exist yet).

    # First reindex so that pentads are fully included. This lets us use coarsen
    # with boundary="exact" to avoid bugs. "trim", for example, could drop data.
    time = monthly.indexes["time"]
    pentad_start = pd.date_range("1956-01-01", "2020-01-01", freq="5YS")
    pentad_stop = pd.date_range("1970-12-31", "2025-12-31", freq="5Y")
    start = pentad_start[pentad_start.searchsorted(time[0], side="left") - 1]
    stop = pentad_stop[pentad_stop.searchsorted(time[-1], side="right") - 1]
    monthly_ = monthly.reindex(
        time=xr.date_range(
            str(start), str(stop), freq="M", calendar=monthly.time.dt.calendar
        )
    )
    reshaped = (
        monthly_.coarsen(time=5 * 12)
        .construct(time=("time_", "month"))
        .assign_coords(month=np.concatenate([np.arange(1, 13)] * 5))
    )
    reshaped["time_"] = reshaped.time.isel(month=0).data
    reshaped = reshaped.rename({"time": "original_time"}).rename({"time_": "time"})

    # calculate anomaly
    anom_ = reshaped - climatology.reindex(
        time=reshaped.time, method="nearest"
    ).reindex(month=reshaped.month)
    anom = (
        anom_.stack({"newtime": ("time", "month")})
        .drop_vars("newtime")
        .swap_dims({"newtime": "original_time"})
        .rename({"original_time": "time"})
    )
    # 3 month centered rolling mean of anomaly
    oni = anom.rolling(time=3, center=True).mean()

    oni.name = "oni"
    oni.attrs["long_name"] = "ONI"
    oni.attrs["standard_name"] = "oceanic_nino_index"

    return oni


def make_enso_transition_mask(oni):
    """
    Make ENSO transition mask following Warner & Moum (2019)

    Parameters
    ----------
    oni: float,
        Oceanic Nino Index time series
    """
    from xarray.core.missing import _get_nan_block_lengths

    # While they say 0.5, 0.45 seems to work better
    thresh = 0.45

    # Need to add an extra value so that things work at the end
    # of the time series
    oni = oni.pad(time=(0, 1), mode="constant", constant_values=0)
    newtime = oni.time.data
    newtime[-1] = newtime[-2] + np.timedelta64(1, "D")
    oni["time"] = newtime

    enso = xr.full_like(
        oni.isel(time=slice(-1)), fill_value="____________", dtype="U12"
    )
    index = oni.indexes["time"] - oni.indexes["time"][0]

    en_mask = (
        _get_nan_block_lengths(
            xr.where(oni >= thresh, np.nan, 0, keep_attrs=False),
            dim="time",
            index=index,
        ).isel(time=slice(-1))
    ) >= pd.Timedelta("59d")

    ln_mask = _get_nan_block_lengths(
        xr.where(oni <= -thresh, np.nan, 0, keep_attrs=False), dim="time", index=index
    ).isel(time=slice(-1)) >= pd.Timedelta("59d")
    # neut_mask = _get_nan_block_lengths(
    # xr.where((ssta < 0.5) & (ssta > -0.5), np.nan, 0),
    # dim="time", index=index)
    # >= pd.Timedelta("120d")

    # donidt = oni.diff("time").reindex(time=oni.time)
    oni = oni.isel(time=slice(-1))
    donidt = oni.diff("time", label="upper").reindex(time=oni.time)
    index = index[:-1]

    # warm_mask = _get_nan_block_lengths(
    # xr.where(donidt >= 0, np.nan, 0, keep_attrs=False),
    # dim="time", index=index) >= pd.Timedelta("59d")
    cool_mask = _get_nan_block_lengths(
        xr.where(donidt <= 0, np.nan, 0, keep_attrs=False), dim="time", index=index
    ) >= pd.Timedelta("120d")

    # Manual fixes to match Warner & Moum figure
    # cool_mask.loc[{"time": "2009-01"}] = True

    warm_mask = ~cool_mask

    # warm_mask = donidt >= 0
    # cool_mask = donidt <= 0
    enso.loc[en_mask & warm_mask] = "El-Nino warm"
    enso.loc[en_mask & cool_mask] = "El-Nino cool"
    enso.loc[ln_mask & warm_mask] = "La-Nina warm"
    enso.loc[ln_mask & cool_mask] = "La-Nina cool"

    # tweaks from Warner & Moum
    try:
        enso.loc[{"time": "2015-12"}] = "El-Nino warm"
        enso.loc[{"time": "2015-01"}] = "El-Nino warm"
    except KeyError:
        pass

    enso.coords["en_mask"] = en_mask
    enso.coords["ln_mask"] = ln_mask
    enso.coords["warm_mask"] = warm_mask
    enso.coords["cool_mask"] = cool_mask

    enso.name = "enso_phase"
    enso.attrs = {
        "description": (
            "Warner & Moum (2019) ENSO transition phase; "
            "El-Nino = ONI > 0.5 for at least 6 months; "
            "La-Nina = ONI < -0.5 for at least 6 months"
        )
    }

    return enso


def normalize_z_da(da):
    """Normalize vertical depth so that positive is always up."""
    with xr.set_options(keep_attrs=True):
        datapos = da.attrs.get("positive", None)
        if not datapos:
            Z = da.cf["Z"]
            if "positive" not in Z.attrs:
                raise ValueError(
                    "Could not find a 'positive' attribute to use for normalization."
                )
            newz = normalize_z(Z)
            da[Z.name] = newz
            return da

        if datapos == "down":
            da = da * -1
            da.attrs["positive"] = "up"

    return da


def normalize_z(obj, sort=False):
    if isinstance(obj, xr.DataArray):
        return normalize_z_da(obj)
    else:
        with xr.set_options(keep_attrs=True):
            out = obj.copy(deep=True)
            for z in set(out.cf.axes["Z"]) & set(out.dims):
                assert "positive" in out[z].attrs
                out[z] = normalize_z_da(out[z])
                if sort:
                    out = out.sortby(z)

        return out


def sel_like(da, other, dims):
    """Select along dims using cf-xarray"""
    if isinstance(dims, str):
        dims = (dims,)
    return da.cf.sel(
        {dim: slice(other.cf[dim].data[0], other.cf[dim].data[-1]) for dim in dims}
    )


def plot_timeseries(tree, var, obs="TAO"):
    """Line hvplot of time series. Selects to match obs."""
    if obs is not None:
        obsts = tree[obs].ds.reset_coords().cf[var]
        title = obsts.attrs["long_name"]
    else:
        title = var
    handles = []
    for name, tree in tree.children.items():
        ds = tree.ds
        kwargs = dict(label=name)
        if name != obs:
            try:
                toplot = ds.reset_coords().cf[var]
                if obs is not None:
                    toplot = toplot.pipe(sel_like, obsts, "time")
            except KeyError:
                raise KeyError(f"Dataset {name!r} is missing {var!r}")
            kwargs.pop("color", None)
        else:
            if obs is None:
                continue
            toplot = obsts
            kwargs["color"] = "darkgray"
        handles.append(toplot.hvplot.line(**kwargs))

    return hv.Overlay(handles).opts(
        legend_position="right",
        frame_width=700,
        title=title,
        xlabel="",
    )


def plot_profile_fill(da, label, **kwargs):
    import hvplot.pandas  # noqa

    assert_z_is_normalized(da.to_dataset())

    prefix = da.name

    Zname = da.cf.axes["Z"][0]
    da = da.copy(deep=True)
    mean = da.mean("time")
    std = da.std("time")

    df_ = (
        xr.Dataset({f"{prefix}_low": mean - std, f"{prefix}_high": mean + std})
        .reset_coords(drop=True)
        .to_dataframe()
    )

    area = df_.hvplot.area(
        x=Zname,
        y=f"{prefix}_low",
        y2=f"{prefix}_high",
        label=label,
        hover=False,
        **kwargs,
    ).opts(alpha=0.1, muted_fill_alpha=0, muted_line_alpha=0)
    line = mean.hvplot.line(
        label=label, muted_line_alpha=0, group_label=label, **kwargs
    ).opts(xrotation=30, xaxis="top")

    return area * line


def cfplot(da, label, **kwargs):
    Zname = da.cf.axes["Z"][0]
    da = da.load().copy(deep=True)
    da[Zname] = normalize_z(da[Zname])
    return da.hvplot.line(label=label, **kwargs)


def map_hvplot(func, datasets, visible=None):
    if visible is None:
        visible = []
    return hv.Overlay(
        [func(ds, name=name, muted=name in visible) for name, ds in datasets.items()]
    ).collate()


def get_mld_tao_theta(theta):
    """
    Given pot temp field, estimate MLD as depth where dθ > 0.15C
    """
    if not isinstance(theta, xr.DataArray):
        raise ValueError(f"Expected DataArray, received {theta.__class__.__name__}")

    thetai = theta.cf.interp(Z=np.arange(0, -200, -1))
    dθ = thetai - thetai.cf.sel(Z=[0, -5], method="nearest").cf.max("Z")

    thresh = xr.where(dθ < -0.1, dθ.cf["Z"], np.nan, keep_attrs=False)
    thresh.attrs = dθ.cf["Z"].attrs
    for dim in thresh.dims:
        thresh[dim].attrs = dθ[dim].attrs
    mld = thresh.cf.max("Z")

    mld.name = "mldT"
    mld.attrs = {
        "long_name": "MLD$_θ$",
        "units": "m",
        "description": (
            "Interpolate θi to 1m grid. " "Search for max depth where " " |dθ| > 0.15"
        ),
    }

    return mld


def add_turbulence_quantities(ds, grid):
    """
    This is an inaccurate estimate of ε, χ.
    We should accumulate online and save ε, χ directly.
    """

    # buoyancy flux
    α = ds.cf["sea_water_thermal_expansion_coefficient"]
    β = ds.cf["sea_water_haline_contraction_coefficient"]
    if set(α.dims) != set(ds.Tz.dims):
        α = xgcm_interp_to(grid, α, axis="Z", to="outer")
    if set(β.dims) != set(ds.Sz.dims):
        β = xgcm_interp_to(grid, β, axis="Z", to="outer")

    K = ds.cf["ocean_vertical_heat_diffusivity"]
    ds["Jb"] = -9.81 * (-K * α * ds.Tz + K * β * ds.Sz)
    ds["Jb"].attrs["standard_name"] = "ocean_vertical_diffusive_buoyancy_flux"

    if "ocean_vertical_diffusive_heat_flux" not in ds.cf.keys():
        ds["Tflx_dia_diff"] = K * ds.Tz
        ds["Tflx_dia_diff"].attrs = {
            "units": "degC m s-1",
            "standard_name": "ocean_vertical_diffusive_heat_flux",
        }

    # define constants, these could be moved up as arguments if we wanted to vary
    # cp = 4186  # J/g°C
    # rho = 1028

    if "Jq" not in ds:
        # Jq
        ds["Jq"] = -1028 * 4186 * ds.cf["ocean_vertical_diffusive_heat_flux"]
        ds["Jq"].attrs = {"units": "W/m^2", "long_name": "$J_q^t$"}

    visc_criteria = {
        "ocean_vertical_x_viscosity": {
            "standard_name": "ocean_vertical_x_viscosity|ocean_vertical_viscosity"
        },
        "ocean_vertical_y_viscosity": {
            "standard_name": "ocean_vertical_y_viscosity|ocean_vertical_viscosity"
        },
    }
    if "eps" not in ds and "eps_chi" not in ds:
        with cfxr.set_options(custom_criteria=visc_criteria):
            shear_prodx = (
                grid.interp(ds.cf["ocean_vertical_x_viscosity"], "Z")
                * grid.derivative(ds.cf["sea_water_x_velocity"], "Z") ** 2
            )
            shear_prody = (
                grid.interp(ds.cf["ocean_vertical_y_viscosity"], "Z")
                * grid.derivative(ds.cf["sea_water_y_velocity"], "Z") ** 2
            )

        ds["shear_prod"] = shear_prodx + xgcm_interp_to(
            grid, shear_prody, axis="Y", to="center"
        )
        ds["shear_prod"].attrs = {"long_name": "$SP$", "units": "W/kg"}

        ds["eps"] = ds["shear_prod"] + ds["Jb"]

    if "chi" not in ds:
        ds["chi"] = 2 * ds.cf["ocean_vertical_heat_diffusivity"] * ds.Tz**2
        ds["chi"].attrs = {"long_name": "$χ$", "units": "C^2/s"}

    if "eps" not in ds and "eps_chi" not in ds:
        ds["eps_chi"] = ds["N2"] * ds.chi / 2 / ds["Tz"] ** 2 / 0.2

    for var in ["eps", "eps_chi", "chi"]:
        if var in ds:
            assert ds[var].ndim == ds.cf["sea_water_x_velocity"].ndim, (
                var,
                ds[var].dims,
            )

    # flux Ri
    ds["Rif"] = ds.Jb / ds.eps
    ds.Rif.attrs["standard_name"] = "flux_richardson_number"


def xgcm_interp_to(grid, da, *, axis, to):
    # TODO: upstream to xgcm
    yaxes = grid.axes[axis]
    try:
        pos = yaxes._get_position_name(da)[0]
    except KeyError:
        print(f"returning {da.name}")
        return da
    if pos != to:
        da = grid.interp(da, axis, to=to)
    return da


def load(ds):
    return ds.update(ds.cf[LOAD_VARNAMES].load())


def load_tree(dt):
    for name, node in dt.children.items():
        dt[name].update(node.ds.cf[LOAD_VARNAMES].load())
    return dt


def persist_tree(dt):
    for name, node in dt.children.items():
        dt[name].update(node.ds.cf[LOAD_VARNAMES].persist())
    return dt


def plot_enso_transition(oni, enso_transition):
    import hvplot.pandas  # noqa

    handles = []
    for phase in np.unique(enso_transition.data):
        handles.append(
            oni.where(enso_transition == phase)
            .reset_coords(drop=True)
            .hvplot.bar(color=tuple(ENSO_COLORS.get(phase, (0.5, 0.5, 0.5))))
            .opts(line_alpha=0, bar_width=1.5)
        )

    return hv.Overlay(handles).opts(frame_width=1200)


def interp_to_center(ds):
    # import toolz as tlz

    # ix0 = np.arange(1, ds.sizes["xq"], 4)
    # ix1 = ix0 + 1
    # indices = list(tlz.interleave([ix0, ix1]))

    # print(ds.xq.data[indices])
    out = (
        ds
        # .isel(xq=[1, 2, 5, 6, 8, 9, 12, 13, 16, 17])
        .isel(xq=[1, 2])
        .coarsen(xq=2)
        .mean()
    )
    out = out.sel(xh=out.xq.data, method="nearest", tolerance=0.05)

    for var in out:
        if "xq" in out[var].dims:
            out[var] = out[var].rename({"xq": "xh"}).drop_vars("xh")
    return out.drop_vars("xq")


def read_mom6_sections(casename):
    from mom6_tools import wright_eos
    from mom6_tools.sections import combine_variables_by_coords, read_raw_files

    dirname = f"/glade/scratch/dcherian/{casename}/run/"
    globstr = f"{dirname}/*TAO*140W*.nc.*"
    files = sorted(glob.glob(globstr))

    if not files:
        raise ValueError(
            f"No files found. Check casename={casename!r} and glob string: {globstr!r}"
        )

    dsets = read_raw_files(files, parallel=True)

    combined = combine_variables_by_coords(dsets)
    combined["time"] = combined["time"] + datetime.timedelta(days=1957 * 365)

    mom6tao = interp_to_center(combined).cf.chunk({"Y": -1})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mom6tao["time"] = mom6tao.indexes["time"].to_datetimeindex()

    mom6tao["dens"] = wright_eos.wright_eos(mom6tao.thetao, mom6tao.so, 0)
    mom6tao["dens"].attrs.update(
        {"units": "kg/m^3", "standard_name": "sea_water_potential_density"}
    )
    mom6tao["densT"] = wright_eos.wright_eos(mom6tao.thetao, 35, 0)
    mom6tao["densT"].attrs.update(
        {"standard_name": "sea_water_potential_density", "units": "kg/m3"}
    )

    mom6tao["α"] = wright_eos.alpha_wright_eos(mom6tao.thetao, mom6tao.so, p=0) / 1025
    mom6tao["α"].attrs = {
        "standard_name": "sea_water_thermal_expansion_coefficient",
        "units": "C-1",
    }
    mom6tao["β"] = wright_eos.beta_wright_eos(mom6tao.thetao, mom6tao.so, p=0) / 1025
    mom6tao["β"].attrs = {
        "standard_name": "sea_water_haline_contraction_coefficient",
        "units": "kg/g",
    }

    if "Kv_v" not in mom6tao:
        warnings.warn("Kv_v not present. Assuming equal to Kv_u")
        mom6tao["Kv_v"] = (mom6tao["vo"].dims, mom6tao.Kv_u.data)

    mom6tao.Kv_u.attrs["standard_name"] = "ocean_vertical_x_viscosity"
    mom6tao.Kv_v.attrs["standard_name"] = "ocean_vertical_y_viscosity"
    mom6tao.Kd_heat.attrs["standard_name"] = "ocean_vertical_heat_diffusivity"
    mom6tao.Tflx_dia_diff.attrs["standard_name"] = "ocean_vertical_diffusive_heat_flux"

    return mom6tao


def mom6_sections_to_zarr(casename):
    mom6tao = read_mom6_sections(casename)
    mom6tao.drop_vars(
        ["average_DT", "average_T2", "average_T1", "time_bnds"], errors="ignore"
    ).chunk({"time": 24 * 365}).to_zarr(
        f"{ROOT}/{casename}/run/tao.zarr",
        mode="w",
        consolidated=True,
    )


def load_mom6_sections(casename, use_reference_files=True):
    import xgcm

    # root = "/glade/scratch/dcherian/archive/"
    mom6tao = xr.open_dataset(
        f"{ROOT}/{casename}/run/tao.zarr",
        engine="zarr",
        chunks="auto",
        consolidated=True,
    )
    # Unfortunately have to do this here so that Grid is right.
    mom6tao = normalize_z(mom6tao, sort=True)

    mom6tao.Kv_u.attrs["standard_name"] = "ocean_vertical_x_viscosity"
    mom6tao.Kv_v.attrs["standard_name"] = "ocean_vertical_y_viscosity"
    mom6tao.Kd_heat.attrs["standard_name"] = "ocean_vertical_heat_diffusivity"
    mom6tao.Tflx_dia_diff.attrs["standard_name"] = "ocean_vertical_diffusive_heat_flux"

    grid = xgcm.Grid(
        mom6tao,
        coords={
            "Z": {"outer": "zi", "center": "zl"},
            "X": {"center": "xh"},
            "Y": {"center": "yh", "left": "yq"},
        },
        periodic=False,
        boundary="fill",
        fill_value=np.nan,
        metrics={("Z",): "h"},
    )

    dirname = f"{ROOT}/{casename}/run"
    if not use_reference_files:
        static = xr.open_dataset(*glob.glob(f"{dirname}/*static*.nc"))
        sfc = xr.open_mfdataset(
            sorted(glob.glob(f"{dirname}/*{casename}*sfc*")),
            coords="minimal",
            data_vars="minimal",
            compat="override",
            use_cftime=True,
            parallel=True,
        )
        sfc.coords.update(static.drop_vars("time"))
    else:
        sfc_reference = f"{dirname}/jsons/sfc.json"
        if not os.path.exists(sfc_reference):
            raise OSError(
                f"Reference files for {casename}, surface fields do not exist."
            )

        fs = fsspec.filesystem("reference", fo=sfc_reference)
        mapper = fs.get_mapper(root="")
        sfc = xr.open_zarr(
            mapper, chunks={"time": 25}, use_cftime=True, consolidated=False
        )
    sfc["time"] = sfc.time + datetime.timedelta(days=365 * 1957)
    # sfc["tos"].attrs["coordinates"] = "geolon geolat"

    sst = sfc.cf["sea_surface_temperature"]

    # Calculate a monthly average sea surface temperature
    # in the Nino 3.4 region (5°S-5°N, 170°W-120°W).
    monthly_ = (
        sst.cf.sel(latitude=slice(-5, 5), longitude=slice(-170, -120))
        .cf.mean(["X", "Y"])
        .resample(time="M")
        .mean()
        .load()
    )

    monthly = monthly_.convert_calendar("gregorian", use_cftime=False)

    mom6tao = prepare(mom6tao, grid, sst_nino34=monthly)

    mom6140 = mom6tao.cf.sel(longitude=-140, latitude=0, method="nearest")
    mom6140 = mom6140.cf.sel(Z=slice(-250, 0))

    # mom6140 = mom6140.update(pdf_N2S2(mom6140))

    return mom6140


def validate_tree(tree):
    """Make sure variables I want to load are present"""

    for name, node in tree.children.items():
        for var in LOAD_VARNAMES:
            if var not in node.ds and var not in node.ds.cf:
                print(f"{var} missing in {name}")


def add_ancillary_variables_microstructure(ds):
    ds = ds.copy()
    ds.u.attrs["standard_name"] = "sea_water_x_velocity"
    ds.v.attrs["standard_name"] = "sea_water_y_velocity"
    ds.pden.attrs["standard_name"] = "sea_water_potential_density"
    ds["S2"] = (
        ds.u.cf.differentiate("Z", positive_upward=True) ** 2
        + ds.v.cf.differentiate("Z", positive_upward=True) ** 2
    )

    # questionable choices to make prepare work
    del ds["zeuc"]
    del ds["sortTbyT"]
    del ds["sortT"]
    ds["densT"] = ds.pden

    assert len(ds.cf.axes["Z"]) == 1
    Zname = ds.cf.axes["Z"][0]
    ds[Zname] = ds[Zname] * -1
    ds[Zname].attrs["positive"] = "up"
    ds = ds.sortby(Zname, ascending=True)

    ds["Tz"] = ds.cf["sea_water_potential_temperature"].cf.differentiate(
        "Z", positive_upward=True
    )
    ds["Sz"] = ds.cf["sea_water_salinity"].cf.differentiate("Z", positive_upward=True)
    ds["N2T"] = -9.81 / 1025 * ds.densT.cf.differentiate("Z", positive_upward=True)
    # ds["Tz"] = (
    # ds.cf["sea_water_potential_temperature"]
    # .cf.differentiate("Z", positive_upward=True)
    # )

    ds = prepare(ds)
    ds.update(pdf_N2S2(ds))
    return ds


def load_microstructure():
    micro = DataTree.from_dict(
        {
            key: xr.open_dataset(f"~/work/datasets/microstructure/osu/{key}.nc")
            for key in ["equix", "tropicheat", "tiwe"]
        }
    )
    micro = micro.map_over_subtree(add_ancillary_variables_microstructure)
    return micro


def load_tao():
    tao_gridded = xr.open_dataset(
        os.path.expanduser("~/work/pump/zarrs/tao-gridded-ancillary.zarr"),
        chunks="auto",
        engine="zarr",
    ).sel(longitude=-140, time=slice("1996", None))
    tao_gridded["pressure"] = gsw_xarray.p_from_z(
        tao_gridded.depth, lat=tao_gridded.latitude
    )
    tao_gridded["SA"] = gsw_xarray.SA_from_SP(
        tao_gridded.S, tao_gridded.pressure, tao_gridded.longitude, tao_gridded.latitude
    )
    tao_gridded["CT"] = gsw_xarray.CT_from_t(
        SA=tao_gridded.SA, t=tao_gridded.T, p=tao_gridded.pressure
    )
    tao_gridded["α"] = gsw_xarray.alpha(
        tao_gridded.SA, tao_gridded.CT, tao_gridded.pressure
    )
    tao_gridded["β"] = gsw_xarray.beta(
        tao_gridded.SA, tao_gridded.CT, tao_gridded.pressure
    )
    tao_gridded.α.attrs["standard_name"] = "sea_water_thermal_expansion_coefficient"
    tao_gridded.β.attrs["standard_name"] = "sea_water_haline_contraction_coefficient"
    tao_gridded["depth"].attrs["axis"] = "Z"
    tao_gridded["shallowest"].attrs.clear()

    tao_gridded["Tz"] = tao_gridded.T.cf.differentiate("Z")
    tao_gridded.Tz.attrs = {"long_name": "$T_z$", "units": "℃/m"}
    tao_gridded["Sz"] = tao_gridded.S.cf.differentiate("Z")
    tao_gridded.Sz.attrs = {"long_name": "$S_z$", "units": "g/kg/m"}

    tao_gridded["swnet"].attrs[
        "standard_name"
    ] = "net_downward_shortwave_flux_at_sea_water_surface"
    tao_gridded["qnet"].attrs[
        "standard_name"
    ] = "surface_downward_heat_flux_in_sea_water"

    tao_gridded["taux"].attrs["standard_name"] = "surface_downward_x_stress"
    tao_gridded["tauy"].attrs["standard_name"] = "surface_downward_y_stress"

    # chi = (dcpy.oceans.read_cchdo_chipod_file(
    #        "~/datasets/microstructure/osu/chipods_0_140W.nc"
    #    )
    #       )

    chi = read_chipod_mat_file(
        os.path.expanduser(
            "~/work/pump/datasets/microstructure/chipods_0_140W_hourly.mat"
        )
    )

    # TAO data are on the hour
    chi["time"] = chi.time - pd.Timedelta("30min")
    chipod = (
        chi  # .sel(time=slice("2015"))
        # "Only χpods between 29 and 69 m are used in this analysis as
        # deeper χpods are more strongly influenced by the
        # variability of zEUC than by surface forcing."
        # - Warner and Moum (2019)
        .reindex(time=tao_gridded.time, method="nearest", tolerance="5min")
        .pipe(normalize_z, sort=True)
        .sel(depth=slice(-90, None))
    )

    chipod = chipod.where(chipod.eps > 1e-14)

    tao_gridded = tao_gridded.update(
        chipod[["chi", "KT", "eps", "Jq"]]
        .reset_coords(drop=True)
        .rename({"depth": "depthchi"})
    )

    tao_gridded = prepare(tao_gridded, oni=process_oni())
    tao_gridded["Tflx_dia_diff"] = tao_gridded.Jq / 1025 / 4200
    tao_gridded["Tflx_dia_diff"].attrs[
        "standard_name"
    ] = "ocean_vertical_diffusive_heat_flux"
    for var in tao_gridded.variables:
        tao_gridded[var].encoding.pop("coordinates", None)

    add_turbulence_quantities(tao_gridded, grid=None)

    return tao_gridded


def load_les_moorings():
    from .les import preprocess_les_dataset

    les_catalog = intake.open_esm_datastore(
        "../catalogs/pump-les-catalog.json",
        read_csv_kwargs={"converters": {"variables": ast.literal_eval}},
    )
    mooring_datasets = les_catalog.search(
        kind="average",
        length="month",
        latitude=0,
    ).to_dataset_dict()
    mooring_datasets = {
        k: prepare(preprocess_les_dataset(v.squeeze()))
        for k, v in mooring_datasets.items()
    }
    moorings = DataTree.from_dict(mooring_datasets)
    return moorings


def solar_decay_moum(h, gamma=0.04):
    # gamma is in /m so h must be in meters
    return 0.45 * np.exp(-gamma * np.abs(h))


def solar_decay_mom(chl, h):
    frac_ir = 0.58
    frac_vis = 0.42
    frac_red = 0.5 * frac_vis
    frac_blue = frac_red

    k_sw_ir = 2.86
    k_sw_red = 0.255
    k_sw_blue = 0.0232

    chi_red = 0.037
    chi_blue = 0.074

    x_red = 0.629
    x_blue = 0.674

    # chl = np.array([1.0e-3, 0.01, 0.1, 1.0, 10.0])
    k_red = k_sw_red + chi_red * (chl**x_red)
    k_blue = k_sw_blue + chi_blue * (chl**x_blue)

    h = np.abs(h)
    frac = (
        frac_ir * np.exp(-k_sw_ir * h)
        + frac_red * np.exp(-k_red * h)
        + frac_blue * np.exp(-k_blue * h)
    )

    # col = ["deepskyblue", "aquamarine", "greenyellow", "green", "darkolivegreen"]

    return frac


def plot_climo_heat_budget_1d(ds, mxldepth=-40, penetration="mom", ax=None):
    if penetration == "mom":
        inputdir = "/glade/p/cesmdata/cseg/inputdata/ocn/mom/tx0.66v1/"
        seawifs = xr.open_dataset(f"{inputdir}/seawifs-clim-1997-2010-tx0.66v1.nc")
        chl_clim = (
            seawifs.cf.sel(latitude=0, longitude=-140, method="nearest")
            .CHL_A.drop_vars("TIME")
            .assign_coords(TIME=np.arange(1, 13))
            .rename({"TIME": "time"})
        )

        pen = (
            solar_decay_mom(h=mxldepth, chl=chl_clim)
            .reindex(time=ds.time.dt.month)
            .assign_coords(time=ds.time.data)
        )
    elif penetration == "moum":
        pen = solar_decay_moum(h=mxldepth)
    else:
        raise ValueError(f"Unknown solar penetration function: {penetration}")

    terms = xr.Dataset()
    terms["Jq_0"] = ds.cf["surface_downward_heat_flux_in_sea_water"]
    terms["Jq_0"].attrs["long_name"] = "$J_q^{-0}$"
    terms["SW"] = ds.cf["net_downward_shortwave_flux_at_sea_water_surface"]
    terms["Jq_h"] = np.abs(
        1025 * 4200 * ds.cf["ocean_vertical_diffusive_heat_flux"]
    ).cf.sel(Z=slice(-60, -20))
    terms["Jq_h"].attrs["long_name"] = "$J_q^{-h}$"
    terms["sst"] = ds.sst
    terms["Ih"] = terms.SW * pen
    terms["Ih"].attrs["long_name"] = "$I^{h}$"
    terms["Jq0-Ih"] = terms.Jq_0 - terms.Ih
    terms["Jq0-Ih"].attrs["long_name"] = "$J_q^0 - I^h$"

    terms = terms.cf.groupby("time.month").mean(["Z", "time"]).load()

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    terms.Jq_h.plot(ax=ax)
    terms["Jq0-Ih"].plot(ax=ax, color="red")
    ax.legend(frameon=False, loc="upper right")
    ax.set_ylim(0, 150)
    ax.set_ylabel(r"[W/m$^2$]")

    ax_sst = ax.twinx()
    terms["sst"].plot(ax=ax_sst, color="k", _labels=False)
    ax_sst.set_ylabel(r"[$^{\circ}$C]")
    ax_sst.set_ylim([24.5, 27.5])
    ax_sst.legend(frameon=False, loc="upper left")

    ax.fill_between(
        range(1, 13),
        terms["Jq0-Ih"],
        terms["Jq_h"],
        where=(terms["Jq0-Ih"] >= terms["Jq_h"]).data,
        facecolor="red",
        alpha=0.2,
        interpolate=True,
    )
    ax.fill_between(
        range(1, 13),
        terms["Jq0-Ih"],
        terms["Jq_h"],
        where=(terms["Jq0-Ih"] < terms["Jq_h"]).data,
        facecolor="blue",
        alpha=0.2,
        interpolate=True,
    )
    ax.grid(b=None)
    ax_sst.grid(b=None)


def read_chipod_mat_file(fname, mask=True):
    import h5py
    import pandas as pd

    file = h5py.File(fname, mode="r")

    chipod = xr.Dataset()

    chipod["depth"] = ("depth", file["chr"]["depth"][:].squeeze())
    chipod["time"] = ("time", file["chr"]["time"][:].squeeze())
    chipod["time"] = pd.to_datetime(chipod.time.data - 719529, unit="D")
    chipod["time"] = chipod.time.dt.round("min")
    for var in ["Jq", "Kt", "N2", "T", "chi", "dTdz", "eps"]:
        chipod[var] = xr.DataArray(file["chr"][var][:, :], dims=("time", "depth"))
    chipod = chipod.cf.guess_coord_axis()
    chipod["depth"].attrs["positive"] = "down"
    chipod = chipod.rename({"Kt": "KT"})
    chipod["KT"].attrs["standard_name"] = "ocean_vertical_heat_diffusivity"

    file.close()

    locs = [
        # from sally; all three depths show an increase though...
        # so I've commented it out
        # {"time": "2007-01-14 10:30:00", "depth": 49.0},
        # From sally
        {"time": "2007-01-28 14:30:00", "depth": 59.0},
        # the rest are mine
        {"time": "2011-02-08 10:30:00", "depth": 49.0},
        {"time": "2010-12-22 05:30:00", "depth": 49.0},
        {"time": "2009-12-21 13:30:00", "depth": 89.0},
    ]

    if mask:
        for loc in locs:
            for var in ["chi", "eps", "Jq", "KT"]:
                chipod[var].loc[loc] = np.nan
    return chipod
