import dcpy
import flox.xarray
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

ENSO_COLORS_RGB = {
    "El-Nino warm": (177, 0, 19),
    "El-Nino cool": (255, 108, 50),
    "La-Nina warm": (55, 150, 202),
    "La-Nina cool": (15, 24, 139),
}
ENSO_COLORS = {k: np.array(v) / 255 for k, v in ENSO_COLORS_RGB.items()}
ENSO_TRANSITION_PHASES = ENSO_COLORS.keys()


# TODO: delete
def ddz(data, grid, h):
    Δ = grid.diff(data, "Z")
    Δz = grid.interp(grid.get_metric(data, "Z"), "Z")
    return (Δ / Δz).rename(f"d{data.name}dz")


def prepare(ds, grid=None, sst_nino34=None):
    dens = ds.cf["sea_water_potential_density"]
    u = ds.cf["sea_water_x_velocity"]
    v = ds.cf["sea_water_y_velocity"]

    out = ds.copy()
    with xr.set_options(arithmetic_join="exact"):
        # dρdz = ddz(dens, grid, ds.h)
        # dudz = ddz(u, grid, ds.hu)
        # dvdz = ddz(v, grid, ds.hv)
        if "N2" not in ds:
            assert grid is not None
            dρdz = grid.derivative(dens, "Z")
            out["N2"] = -9.81 / grid.interp_like(dens, like=dρdz) * dρdz
            if u.cf["Z"].attrs.get("positive", None) == "down":
                out["N2"] *= -1

            out["N2"].attrs["long_name"] = "$N^2$"
            out["N2"].attrs["units"] = "s$^{-2}$"
        else:
            out["N2"] = ds.N2

        if "S2" not in ds:
            assert grid is not None
            dudz = grid.derivative(u, "Z")
            dvdz = grid.derivative(v, "Z")
            out["S2"] = dudz**2 + grid.interp(dvdz, "Y", to="center") ** 2
            out["S2"].attrs["long_name"] = "$S^2$"
            out["S2"].attrs["units"] = "s$^{-2}$"
        else:
            out["S2"] = ds.S2

        out["shred2"] = out.S2 - 4 * out.N2
        out.shred2.attrs["long_name"] = "$Sh_{red}^2$"
        out.shred2.attrs["units"] = "$s^{-2}$"

        out["Ri"] = out.N2 / out.S2
        out.Ri.attrs["long_name"] = "Ri"
        out.Ri.attrs["units"] = ""

    # out["eucmax"] = pump.calc.get_euc_max(u)
    out["eucmax"] = euc_max(
        u.cf.sel(Z=slice(10, 350)).cf.sel(Y=0, method="nearest", drop=True)
    )

    if sst_nino34 is not None:
        oni = calc_oni(sst_nino34)
        enso_transition = make_enso_transition_mask(oni).rename("enso_transition")
        out.coords.update(
            xr.merge([oni, enso_transition]).reindex(time=ds.time, method="ffill")
        )

    return out


def euc_max(u):
    euc_max = u.cf.idxmax("Z")
    euc_max.attrs.clear()
    euc_max.attrs["units"] = "m"
    if "positive" in u.cf["Z"].attrs:
        euc_max.attrs["positive"] = u.cf["Z"].attrs["positive"]
    return euc_max


def find_pdf_contour(density, targets):
    """
    Finds PDF contour that encloses target fraction of the samples.
    """

    from typing import Iterable

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


def to_density(counts, dims=("N2_bins", "S2_bins")):
    total = counts.sum(dims)
    area = 1
    for dim in dims:
        area = area * counts[dim].copy(
            data=np.array([i.right - i.left for i in counts[dim].data])
        )
    density = counts / total / area
    density.coords["bin_areas"] = area
    return density


def pdf_N2S2(data, coord_is_center=False):
    """
    Calculate pdf of data in S2-4N2 space.
    """

    bins = np.linspace(-5, -2, 30)
    index = pd.IntervalIndex.from_breaks(bins)

    if np.all(data.cf["Z"].data < 1):
        data["S2"] = data.S2.where(data.cf["Z"] > (data.eucmax + 5))
    else:
        data["S2"] = data.S2.where(data.cf["Z"] < (data.eucmax - 5))

    by = [np.log10(4 * data.N2), np.log10(data.S2)]
    expected_groups = [index, index]
    isbin = [True, True]

    if "enso_transition" in data.variables:
        by.append(data.enso_transition)
        expected_groups.append(None)
        isbin.append(False)

    enso_counts = flox.xarray.xarray_reduce(
        data.S2, *by, func="count", expected_groups=tuple(expected_groups), isbin=isbin
    )

    to_concat = []
    if "enso_transition" in data.variables:
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
    counts.N2_bins.attrs["long_name"] = "log$_{10} 4N^2$"
    counts.S2_bins.attrs["long_name"] = "log$_{10} S^2$"

    if "enso_transition" in data.variables:
        counts["enso_transition_phase"].attrs = {
            "description": (
                "El-Nino transition phase as defined by Warner and Moum (2019)."
                " 'none' uses all available data. 'all' sums over all ENSO transition phases"
            )
        }

    counts.attrs = {"long_name": "$P(S^2, 4N^2)$"}
    with xr.set_options(keep_attrs=True):
        density = to_density(counts, dims=("N2_bins", "S2_bins"))

    if coord_is_center:
        vec = np.linspace(-5, -2, 50)
        density["N2_bins"] = (vec[1:] + vec[:-1]) / 2
        density["S2_bins"] = (vec[1:] + vec[:-1]) / 2

    return density


def plot_n2s2pdf(da, targets=(0.5, 0.75), pcolor=True, **kwargs):
    da = da.squeeze()
    levels = find_pdf_contour(da, targets=targets)
    if pcolor:
        da.plot(robust=True, **kwargs)
    cs = da.reset_coords(drop=True).plot.contour(levels=levels, **kwargs)
    dcpy.plots.line45(ax=kwargs.get("ax", None), ls="--", lw=1.25)
    return cs


def hvplot_n2s2pdf(da, targets=(0.5, 0.75), pcolor=True, **kwargs):
    import holoviews as hv
    import hvplot.xarray  # noqa

    da = da.squeeze().copy()
    da["N2_bins"] = pd.IntervalIndex(da.N2_bins.data).mid.to_numpy()
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
    for label, group in ds.n2s2pdf.groupby(hue):
        group["N2_bins"] = group["N2_bins"].copy(
            data=pd.IntervalIndex(group.N2_bins.data).mid
        )
        group["S2_bins"] = group["S2_bins"].copy(data=group.N2_bins.data)

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
    reshaped = (
        monthly.coarsen(time=5 * 12, boundary="trim")
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

    return oni


def make_enso_transition_mask(oni):
    """
    Make ENSO transition mask following Warner & Moum (2019)

    Parameters
    ----------
    oni: float,
        Oceanic Nino Idnex time series
    """
    from xarray.core.missing import _get_nan_block_lengths

    enso = xr.full_like(oni, fill_value="____________", dtype="U12")
    index = oni.indexes["time"] - oni.indexes["time"][0]
    en_mask = _get_nan_block_lengths(
        xr.where(oni >= 0.45, np.nan, 0, keep_attrs=False), dim="time", index=index
    ) >= pd.Timedelta("59d")

    ln_mask = _get_nan_block_lengths(
        xr.where(oni <= -0.5, np.nan, 0, keep_attrs=False), dim="time", index=index
    ) >= pd.Timedelta("59d")
    # neut_mask = _get_nan_block_lengths(xr.where((ssta < 0.5) & (ssta > -0.5), np.nan, 0), dim="time", index=index) >= pd.Timedelta("120d")

    # donidt = oni.diff("time").reindex(time=oni.time)
    donidt = oni.differentiate("time")

    # warm_mask = _get_nan_block_lengths(xr.where(donidt >= 0, np.nan, 0, keep_attrs=False), dim="time", index=index) >= pd.Timedelta("59d")
    cool_mask = _get_nan_block_lengths(
        xr.where(donidt <= 0, np.nan, 0, keep_attrs=False), dim="time", index=index
    ) >= pd.Timedelta("120d")
    warm_mask = ~cool_mask

    # warm_mask = donidt >= 0
    # cool_mask = donidt <= 0
    enso.loc[en_mask & warm_mask] = "El-Nino warm"
    enso.loc[en_mask & cool_mask] = "El-Nino cool"
    enso.loc[ln_mask & warm_mask] = "La-Nina warm"
    enso.loc[ln_mask & cool_mask] = "La-Nina cool"

    enso.coords["warm_mask"] = warm_mask
    enso.coords["cool_mask"] = cool_mask

    enso.name = "enso_phase"
    enso.attrs[
        "description"
    ] = "Warner & Moum (2019) ENSO transition phase; El-Nino = ONI > 0.5 for at least 6 months; La-Nina = ONI < -0.5 for at least 6 months"

    return enso
