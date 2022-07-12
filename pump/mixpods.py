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

    assert np.all(data.cf["Z"].data < 1)
    data["S2"] = data.S2.where(data.cf["Z"] > data.eucmax + 5)

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
    levels = find_pdf_contour(da, targets=targets)
    if pcolor:
        da.plot(robust=True)
    cs = da.reset_coords(drop=True).plot.contour(levels=levels, **kwargs)
    dcpy.plots.line45(ax=kwargs.get("ax", None))
    return cs


def plot_enso_stability_diagram(ds, title=None, ax=None, add_legend=True, **kwargs):
    if ax is None:
        ax = plt.gca()

    handles = []
    labels = []
    for label, group in ds.n2s2pdf.groupby("enso_transition_phase"):
        group["N2_bins"] = group["N2_bins"].copy(
            data=pd.IntervalIndex(group.N2_bins.data).mid
        )
        group["S2_bins"] = group["S2_bins"].copy(data=group.N2_bins.data)

        if "Nin" not in label:
            continue

        color = kwargs.pop("color", ENSO_COLORS[label])
        cs = plot_n2s2pdf(
            group.squeeze(),
            targets=[
                0.5,
            ],
            colors=(color,),
            pcolor=False,
            **kwargs,
            ax=ax
        )
        handles.append(cs.legend_elements()[0][0])
        labels.append(label)

    if add_legend:
        ax.legend(handles, labels, bbox_to_anchor=(1, 1))
    ax.set_xlim([-4.5, -2.5])
    ax.set_ylim([-4.5, -2.5])
    if title:
        ax.set_title(title)


def plot_stability_diagram_by_dataset(tao_gridded, simulations, fig=None):
    if fig is None:
        fig = plt.figure(constrained_layout=True, figsize=(8, 4))

    ax = fig.subplots(1, 1 + len(simulations), sharex=True, sharey=True)

    plot_enso_stability_diagram(
        tao_gridded, ax=ax[0], linewidths=2, title="TAO", add_legend=False
    )
    for name, sim in simulations.items():
        plot_enso_stability_diagram(sim, linewidths=2, ax=ax[1], title=name)

    dcpy.plots.clean_axes(ax)


def plot_stability_diagram_by_phase(tao_gridded, simulations, fig=None):
    if fig is None:
        fig = plt.figure(figsize=(12, 4), constrained_layout=True)

    axx = fig.subplots(1, 4, sharex=True, sharey=True)

    ax = dict(zip(ENSO_TRANSITION_PHASES, axx))

    for phase in ENSO_TRANSITION_PHASES:
        kwargs = dict(ax=ax[phase], add_legend=False)
        plot_enso_stability_diagram(
            tao_gridded.sel(enso_transition_phase=[phase]),
            **kwargs,
            color="k",
            linewidths=2
        )

        for name, sim in simulations.items():
            plot_enso_stability_diagram(
                sim.sel(enso_transition_phase=[phase]), **kwargs
            )

        ax[phase].text(
            x=0.35, y=0.03, s=phase.upper() + "ING", transform=ax[phase].transAxes
        )

    dcpy.plots.clean_axes(axx)
