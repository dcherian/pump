import itertools

import dask
import dcpy.plots
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import xarray
import xarray as xr

from .calc import calc_kpp_terms, calc_reduced_shear, get_dcl_base_Ri, get_mld
from .mdjwf import dens


def fix_gradient_edge_labels(obj, dim):
    diff = obj[dim].diff(dim).data / 2
    diff[1:-1] = 0
    diff[-1] *= -1
    diff = np.insert(diff, -2, 0)
    return obj.assign_coords({dim: obj[dim].data + diff})


cmaps = {
    "S2": dict(cmap=mpl.cm.GnBu, norm=mpl.colors.LogNorm(1e-5, 5e-4)),
    "Jq": dict(
        cmap=mpl.cm.Blues_r,
        vmax=0,
        vmin=-400,
    ),
    "Ri": dict(cmap=mpl.cm.RdGy_r, center=0.5, norm=mpl.colors.Normalize(0, 1)),
    "N2": dict(cmap=mpl.cm.GnBu, norm=mpl.colors.LogNorm(1e-5, 5e-4)),
    "shred2": dict(
        norm=mpl.colors.TwoSlopeNorm(vcenter=-5e-7, vmin=-5e-4, vmax=1e-4),
        cmap=mpl.cm.RdBu_r,
    ),
}


@xarray.plot.dataset_plot._dsplot
def quiver(ds, x, y, ax, u, v, **kwargs):
    from xarray import broadcast

    if x is None or y is None or u is None or v is None:
        raise ValueError("Must specify x, y, u, v for quiver plots.")

    # matplotlib autoscaling algorithm
    scale = kwargs.pop("scale", None)
    if scale is None:
        npts = ds.dims[x] * ds.dims[y]
        # crude auto-scaling
        # scale is typical arrow length as a multiple of the arrow width
        scale = (
            1.8 * ds.to_array().median().values * np.maximum(10, np.sqrt(npts))
        )  # / span

    ds = ds.squeeze()
    x, y, u, v = broadcast(ds[x], ds[y], ds[u], ds[v])

    add_guide = kwargs.pop("add_guide", None)
    kwargs.pop("cmap_params")
    kwargs.pop("hue")
    kwargs.pop("hue_style")
    hdl = ax.quiver(x.values, y.values, u.values, v.values, scale=scale, **kwargs)

    # units = u.attrs.get("units", "")
    # ax.quiverkey(
    #     hdl,
    #     X=0.1,
    #     Y=0.95,
    #     U=median,
    #     label=f"{median}\n{units}",
    #     labelpos="E",
    #     coordinates="figure",
    # )

    return hdl


xarray.plot.dataset_plot.quiver = quiver


def plot_depths(ds, ax=None, **kwargs):

    if ax is None:
        ax = plt.gca()

    if "euc_max" in ds:
        ds.euc_max.plot.line(ax=ax, color="k", lw=1, _labels=False, **kwargs)

    if "dcl_base" in ds:
        ds.dcl_base.plot.line(ax=ax, color="gray", lw=1, _labels=False, **kwargs)

    if "mld" in ds:
        ds.mld.plot.line(ax=ax, color="k", lw=0.5, _labels=False, **kwargs)


def plot_bulk_Ri_diagnosis(ds, f=None, ax=None, buoy=True, **kwargs):
    """
    Estimates fractional contributions of various terms to bulk Richardson
    number.
    """

    def plot_ri_contrib(ax1, ax2, v, factor=1, **kwargs):
        # Better to call differentiate on log-transformed variable
        # This is a nicer estimate of the gradient and is analytically equal
        per = factor * np.log(np.abs(v)).compute().differentiate("longitude")

        # assign one-sided derivative to mid point
        per = fix_gradient_edge_labels(per, "longitude")

        # hdl = per.plot(
        #    ax=ax2,
        #    x="longitude",
        #    label=f"{factor}/{v.name} $∂_x${v.name}",
        #    add_legend=False,
        #    **kwargs,
        # )

        basefmt = "gray"
        markerfmt = f"{kwargs['color']}{kwargs['marker']}"
        hdl = ax2.stem(
            per.longitude,
            per,
            label=f"{factor}/{v.name} $∂_x${v.name}",
            basefmt=basefmt,
            markerfmt=markerfmt,
            # **kwargs,
        )

        v.plot(ax=ax1, x="longitude", **kwargs)
        ax1.set_xlabel("")
        ax1.set_title("")

        return per, hdl

    naxes = 7 if buoy else 5
    if f is None and ax is None:
        f, axx = plt.subplots(
            naxes,
            1,
            constrained_layout=True,
            sharex=True,
            gridspec_kw={"height_ratios": [1] * (naxes - 1) + [2]},
        )
        if buoy:
            names = ["Rib", "h", "du", "db", "u", "b", "contrib"]
        else:
            names = ["Rib", "h", "du", "u", "contrib"]
        ax = dict(zip(names, axx))
        add_legend = True
    else:
        add_legend = False

    colors = dict(
        {
            "us": "C0",
            "ueuc": "C1",
            "bs": "C0",
            "beuc": "C1",
            "Rib": "k",
            "h": "C1",
            "du": "C2",
            "db": "C3",
        }
    )

    factor = dict(zip(ax.keys(), [1, 1, -2, 1]))
    rhs = fix_gradient_edge_labels(xr.zeros_like(ds.bs), "longitude")
    per = dict()
    for var in ax.keys():
        if var not in ["u", "b", "contrib"]:
            per[var], hdl = plot_ri_contrib(
                ax[var],
                ax["contrib"],
                ds[var],
                factor[var],
                color=colors[var],
                **kwargs,
            )
            if var != "Rib":
                rhs += per[var]
            else:
                ri = per[var]
                if "marker" not in kwargs:
                    hdl[0].set_marker("o")

    fields = ["u", "b"] if buoy else ["u"]
    for vv in fields:
        for vvar in ["s", "euc"]:
            var = vv + vvar
            if vvar == "euc":
                factor = -1
                prefix = "-"
            else:
                factor = 1
                prefix = ""
            (factor * ds[var].compute().differentiate("longitude")).plot(
                ax=ax[vv],
                label=f"{prefix}$∂_x {vv}_{{{vvar}}}$",  # label=f'$∂_x{vv}_{{{vvar}}}$',
                color=colors[var],
                **kwargs,
            )
            if add_legend:
                ax[vv].legend(ncol=2)

            dcpy.plots.liney(0, ax[vv])
            ax[vv].set_title("")
            ax[vv].set_xlabel("")
            ax[vv].set_ylabel("")

    ax["u"].set_ylim([-0.04, 0.04])
    ax["du"].set_ylim([-1.3, -0.3])
    ax["u"].set_ylabel("$∂_x u$ [1/s]")

    if buoy:
        ax["b"].set_ylim([-0.0007, 0.0007])
        ax["db"].set_ylim([0.005, 0.05])

    ax["Rib"].set_ylabel("Ri$_b =  Δbh/Δu²$")
    ax["Rib"].set_yscale("log")
    ax["Rib"].set_yticks([0.25, 0.5, 1, 5, 10])
    ax["Rib"].grid(True)

    if buoy:
        # this check makes no sense if buoyancy terms are ignored
        rhs.plot(
            ax=ax["contrib"],
            x="longitude",
            color="C0",
            label="RHS",
            **kwargs,
            add_legend=False,
        )
    if add_legend:
        ax["contrib"].legend(ncol=5)
        dcpy.plots.liney(0, ax=ax["contrib"])
    ax["contrib"].set_ylabel("Fractional changes")
    ax["contrib"].set_title("")
    # ax["contrib"].set_ylim([-0.15, 0.1])

    name = ds.attrs["name"]
    if add_legend:
        ax["Rib"].set_title(f"latitude = 0, {name} dataset")
    else:
        ax["Rib"].set_title(f"latitude = 0")

    f.set_size_inches(8, 10)

    # xr.testing.assert_allclose(ri, rhs)
    return f, ax


def plot_jq_sst(
    model, lon, periods, lat=0, period=None, full=None, eucmax=None, time_Ri=None
):

    tao = None
    if not isinstance(model, xr.Dataset):
        sst = model.surface.theta.sel(longitude=lon, method="nearest")
    else:
        sst = model.sst
        tao = model
        full = model

    if isinstance(periods, slice):
        tperiod = periods
    else:
        if period is None:
            period = model.full.period.sel(longitude=lon, method="nearest")

        tperiod = sst.time.where(period.isin(periods), drop=True)[[0, -1]]
        tperiod = slice(*list(tperiod.values))

    if time_Ri is None:
        time_Ri = {la: tperiod for la in np.atleast_1d(lat)}

    for la in np.atleast_1d(lat):
        if la not in time_Ri:
            time_Ri[la] = tperiod

    if tao is None:
        if full is None:
            tao = model.tao.sel(longitude=lon, time=tperiod, depth=slice(0, -500))
        else:
            tao = full.sel(longitude=lon, method="nearest").sel(
                time=tperiod, depth=slice(0, -500)
            )

    if eucmax is None:
        eucmax = tao.eucmax
    else:
        eucmax = eucmax.sel(longitude=lon, method="nearest")

    eucmax = eucmax.sel(time=tperiod)

    full = full.sel(time=tperiod)
    tao = tao.sel(time=tperiod)
    lat = np.atleast_1d(lat)

    region = dict(latitude=lat, method="nearest")

    mld = tao.mld
    dcl_base = tao.dcl_base
    # mld = get_mld(tao.dens.sel(**region))
    # dcl_base = get_dcl_base_Ri(tao.sel(**region), mld, eucmax)

    mld, dcl_base = dask.compute(mld, dcl_base)
    # tao = model.full.sel(longitude=lon, time=tperiod, depth=slice(0, -500))

    f, ax = plt.subplots(
        2 + len(lat),
        2,
        sharex="col",
        sharey="row",
        constrained_layout=True,
        # gridspec_kw={"height_ratios": [2] * (lenlat1;5)},
        gridspec_kw={"width_ratios": [4, 1]},
    )
    width = dcpy.plots.pub_fig_width("jpo", "two column")
    f.set_size_inches((width, 8))
    f.set_constrained_layout_pads(wspace=0, w_pad=0, h_pad=0, hspace=0)

    ax[0, 1].remove()
    ax[1, 1].remove()

    # First SST
    plt.sca(ax[0, 0])
    # cax = dcpy.plots.cbar_inset_axes(ax[0, 0])
    sst.name = "SST"
    sst.attrs["units"] = "°C"
    sst.cf.sel(time=tperiod, latitude=slice(-2, 5)).plot(
        x="time",
        cmap=mpl.cm.RdYlBu_r,
        robust=True,
        ax=ax[0, 0],
        cbar_kwargs={"label": ""},
    )
    ax[0, 0].set_title(f"{np.abs(np.round(lon, 0))}°W")
    # ax[0, 0].set_title("")
    ax[0, 0].set_xlabel("")

    # Jq
    jq_kwargs = dict(
        x="time",
        **cmaps["Jq"],
        levels=np.arange(-400, 0, 50),
        #    cmap=mpl.cm.Blues_r,
    )
    # cax = dcpy.plots.cbar_inset_axes(ax[1, 0])

    (
        (
            tao.Jq.where(tao.depth < tao.mld)
            .sel(depth=slice(-100), time=tperiod, latitude=slice(-2, 5))
            .sum("depth")
            / 50
        ).plot.contourf(
            ax=ax[1, 0],
            **jq_kwargs,
            add_colorbar=True,
            cbar_kwargs={
                "orientation": "horizontal",
                "shrink": 0.8,
                "aspect": 40,
                "label": "",  # "$J_q^t$ [W/m²]",
                # "ticks": [-600, -400, -200, -100, -50, 0],
                # "cax": cax,
            },
        )
    )
    ax[1, 0].set_xlabel("time")
    ax[1, 0].set_title("")

    dcpy.plots.liney(
        tao.sel(**region).latitude, ax=ax[:2, 0], ls="--", color="k", lw=1, zorder=10
    )

    dcpy.plots.concise_date_formatter(ax[1, 0], show_offset=False)
    [tt.set_visible(True) for tt in ax[1, 0].get_xticklabels()]

    # Jq with eucmax, MLD, DCL
    doRi = True
    if doRi:
        for index, (la, axis, axRi) in enumerate(zip(lat[::-1], ax[2:, 0], ax[2:, 1])):
            tRi = time_Ri[la]
            region = {"latitude": la, "method": "nearest"}
            plt.sca(axis)
            jq = (
                tao.sel(**region).Jq.rolling(depth=2, min_periods=1, center=True).mean()
            )
            jq.plot.contourf(**jq_kwargs, add_colorbar=False)

            rii = (
                tao.Ri.sel(time=tRi)
                .where((tao.depth < mld) & (tao.depth > dcl_base))
                .sel(**region)
                .chunk({"time": -1})
            )
            rii = rii.where(rii.count("time") > 15)
            ri_q = (
                rii.chunk({"time": -1}).quantile(dim="time", q=[0.25, 0.5, 0.75])
            ).compute()

            # .plot.line(ax=axRi, xscale="log", hue="quantile", y="depth", xlim=(0.1, 2))

            # mark Ri distribution time
            t = pd.date_range(tRi.start, tRi.stop, freq="D")
            axis.plot(t, -90 * np.ones(t.shape), color="r", lw=4)

            dcpy.plots.fill_between(
                ri_q.sel(quantile=[0.25, 0.75]),
                axis="x",
                x="quantile",
                y="depth",
                ax=axRi,
                color="r",
                alpha=0.2,
            )
            ri_q.sel(quantile=0.5).plot.line(
                ax=axRi, xscale="linear", y="depth", color="r"
            )
            axRi.set_xlim((0.1, 1))
            dcpy.plots.linex([0.25, 0.4], ax=axRi, lw=1)
            axRi.set_title("")
            axRi.set_ylabel("")
            axRi.set_xlabel("")

            hdl = dcl_base.sel(**region).plot(x="time", color="k", _labels=False)
            dcpy.plots.annotate_end(hdl[0], "$z_{Ri}$", va="top")
            hdl = mld.sel(**region).plot(x="time", color="C1", _labels=False)
            dcpy.plots.annotate_end(hdl[0], "$z_{MLD}$", va="bottom")
            if eucmax is not None and np.abs(la) < 2:
                hdl = eucmax.plot(x="time", color="k", linestyle="--", _labels=False)
                dcpy.plots.annotate_end(hdl[0], "$z_{EUC}$")

            axis.set_ylim([-120, 0])
            axis.set_title(f"latitude={la}°N")
            axis.set_xlabel("")

            if index == 0:
                ax2 = axRi.secondary_xaxis("top")
                ax2.set_xticks([0.25, 0.4])
                ax2.set_xticklabels([0.25, 0.4])
                ax2.tick_params(labeltop=True)
                ax2.tick_params(axis="x", rotation=40)
                [tt.set_horizontalalignment("left") for tt in ax2.get_xticklabels()]

            else:
                axRi.set_xticks([0.25, 0.5, 0.75, 1])
                axRi.tick_params(axis="x", rotation=40)
                [tt.set_horizontalalignment("right") for tt in axRi.get_xticklabels()]

            # axis.text(0.03, 0.05, f"latitude={la}°N", color="k", transform=axis.transAxes)

    ax[1, 1].set_xlabel("")

    # Just tiw phase
    # plt.sca(ax[-1])
    # model.full.tiw_phase.sel(longitude=lon, method="nearest").plot(_labels=False)
    # [aa.set_xlabel("") for aa in ax[:-1]]

    # for aa in [ax[0, 0], ax[2, 1]]:
    #     [tt.set_visible(False) for tt in aa.get_xticklabels()]
    # [tt.set_visible(True) for tt in ax[0, 0].get_xticklabels()]
    ax[1, 0].tick_params(labelbottom=True)
    # dcpy.plots.concise_date_formatter(ax[-1, 0], show_offset=False, minticks=9)
    dcpy.plots.concise_date_formatter(ax[1, 0], show_offset=True, minticks=9)
    return f, ax


def plot_debug_sst_front(model, lon, periods):

    _, sst, sstfilt, gradT, tiw_phase, period, _ = model.get_quantities_for_composite(
        longitudes=[lon]
    )

    f, ax = plt.subplots(3, 1, sharex=True)

    sst.sel(longitude=lon, method="nearest").plot(
        x="time", ax=ax[0], add_colorbar=False, vmin=22, vmax=27, cmap=mpl.cm.Spectral_r
    )

    gradT_subset = (
        gradT.sel(longitude=[lon])
        .assign_coords(period=period, tiw_phase=tiw_phase)
        .squeeze()
    )
    gradT_subset = gradT_subset.where(
        gradT_subset.period.isin(periods), drop=True
    ).load()

    gradT_subset.plot(x="time", robust=True, ax=ax[1], add_colorbar=False)
    dcpy.plots.liney([0, 1, 2, 3, 4, 5], ax=ax[:2], color="w", zorder=10)

    (gradT_subset.sel(latitude=slice(0, 2)).mean("latitude") * 10).plot(
        x="time", ax=ax[-1]
    )
    gradT_subset.tiw_phase.plot(x="time", ax=ax[-1].twinx(), color="k", _labels=False)

    dcpy.plots.linex(
        gradT_subset.tiw_phase.where(
            gradT_subset.tiw_phase.isin([90, 180]), drop=True
        ).time,
        ax=ax,
        zorder=10,
        color="gray",
    )

    dcpy.plots.linex(
        gradT_subset.tiw_phase.where(gradT_subset.tiw_phase.isin([0]), drop=True).time,
        ax=ax,
        zorder=10,
        lw=1,
        color="k",
    )

    (
        (sstfilt.sel(longitude=lon) * 10)
        .where(period.sel(longitude=lon).isin(periods), drop=True)
        .plot(color="C1", ax=ax[-1])
    )

    ax[0].set_ylim([-5, 6])
    ax[1].set_ylim([-5, 6])
    f.set_size_inches((8, 8))


# def plot_shear_terms(shear, dcl=None):
#     kwargs = dict(
#         col="term",
#         x="time",
#         robust=True,
#         cbar_kwargs={
#             "orientation": "horizontal",
#             "shrink": 0.6,
#             "aspect": 40,
#             "pad": 0.15,
#         },
#         vmin=-5e-8,
#         vmax=5e-8,
#         cmap=mpl.cm.RdBu_r,
#     )
#     if "depth" in shear.dims:
#         kwargs["row"] = "depth"

#     fg = shear.sel(latitude=slice(-3, 5)).to_array("term").plot(size=5, **kwargs)
#     if "name" in shear.attrs:
#         fg.fig.suptitle(shear.attrs["name"], y=1.01)

#     def plot():
#         dcl.plot.contour(
#             levels=7, colors="k", robust=True, x="time", add_labels=False, linewidths=1
#         )

#     if dcl is not None:
#         fg.map(plot)


def plot_shred2_time_instant(tsub, ax, add_colorbar):

    kwargs = dict(
        # vmin=-0.02,
        # vmax=0.02,
        **cmaps["shred2"],
        add_colorbar=False,
    )

    fmt = mpl.ticker.ScalarFormatter()
    fmt.set_powerlimits((-1, 1))

    Ric = 0.4
    (tsub.uz ** 2 - 1 / Ric / 2 * tsub.N2).plot(ax=ax["u"], **kwargs)
    (tsub.vz ** 2 - 1 / Ric / 2 * tsub.N2).plot(ax=ax["v"], **kwargs)
    # tsub.v.plot.contour(ax=ax["v"], colors="k", linewidths=1, levels=7)

    hdl = (tsub.uz ** 2 + tsub.vz ** 2 - 1 / Ric * tsub.N2).plot(ax=ax["Ri"], **kwargs)

    if add_colorbar:
        plt.gcf().colorbar(
            hdl,
            ax=[ax["u"], ax["v"], ax["Ri"]],
            orientation="horizontal",
            shrink=0.8,
            aspect=37,
            extend="both",
            label="$Sh_{red}²$ [s$^{-2}$]",
            format=fmt,
        )

    # tsub.u.plot.contour(ax=ax["Ri"], colors="k", linewidths=2, levels=5)
    # tsub.v.plot.contour(ax=ax["Ri"], colors="k", linewidths=1, levels=7)

    # tsub.Ri.plot(
    #    ax=ax["Ri"],
    #    cmap=mpl.cm.PuBu,
    #    vmin=0.1,
    #    vmax=0.5,
    # norm=mpl.colors.LogNorm(0.1, 0.5),
    #    add_colorbar=add_colorbar,
    #    cbar_kwargs={"orientation": "horizontal"} if add_colorbar else None,
    # )

    tsub.Jq.rolling(depth=2, center=True, min_periods=1).mean().plot(
        ax=ax["Jq"],
        cmap=mpl.cm.Blues_r,
        vmin=-250,
        vmax=0,
        add_colorbar=add_colorbar,
        cbar_kwargs={
            "orientation": "horizontal",
            "ax": [ax["Jq"]],
            "label": "$J_q^t$ [W/m²]",
        }
        if add_colorbar
        else None,
    )

    def annotate(tsub, ax):
        tsub.dcl_base.plot(ax=ax, color="w", lw=3, _labels=False)
        tsub.dcl_base.plot(ax=ax, color="k", lw=1, _labels=False)
        tsub.mld.plot(ax=ax, color="w", lw=3, _labels=False)
        tsub.mld.plot(ax=ax, color="orange", lw=1, _labels=False)
        # dcpy.plots.liney(tsub.eucmax, ax=ax, zorder=10, color="k")

    axx = list(ax.values())
    # mask = xr.where((tsub.depth > tsub.mld) | (tsub.depth < tsub.dcl_base), 1, np.nan)
    # [dcpy.plots.plot_mask(aa, mask) for aa in axx]

    [
        aa.set_title(tt)
        for aa, tt in zip(
            axx,
            [
                "$J_q^t$",
                "$u_z^2 + v_z^2- N²/Ri_c$",
                "$u_z^2 - N²/2/Ri_c$",
                "$v_z^2 - N²/2/Ri_c$",
            ],
        )
    ]
    [aa.set_ylabel("") for aa in axx[1:]]
    [aa.set_xticks(np.arange(-5, 6, 1)) for aa in axx]
    [
        aa.tick_params(
            top=False,
        )
        for aa in axx
    ]
    [aa.tick_params(labelbottom=False) for aa in axx]
    [annotate(tsub, aa) for aa in axx]


def plot_tiw_lat_lon_summary(subset, times):

    f = plt.figure(constrained_layout=True)
    width = dcpy.plots.pub_fig_width("jpo", "two column")
    f.set_size_inches((width, 5))

    # gsparent = f.add_gridspec(2, 1, height_ratios=[1, 1.5])
    # left = gsparent[0].subgridspec(2, 3)
    # ax = dict()
    # ax["sst"] = f.add_subplot(left[0, 0])
    # ax["dcl"] = f.add_subplot(left[0, 1], sharex=ax["sst"], sharey=ax["sst"])
    # ax["shred2"] = f.add_subplot(left[0, 2], sharex=ax["sst"], sharey=ax["sst"])
    # ax["uz"] = f.add_subplot(left[1, 0], sharex=ax["sst"], sharey=ax["sst"])
    # ax["vz"] = f.add_subplot(left[1, 1], sharex=ax["sst"], sharey=ax["sst"])
    # ax["N2"] = f.add_subplot(left[1, 2], sharex=ax["sst"], sharey=ax["sst"])

    # axtop = np.array(list(ax.values())).reshape((2, 3))

    f, axtop = plt.subplots(2, 3, sharex=True, sharey=True, constrained_layout=True)

    ax = dict(zip(("sst", "dcl", "shred2", "uz", "vz", "N2"), axtop.flat))
    # cax_sst = inset_axes(ax["sst"], **inset_kwargs, bbox_transform=ax["sst"].transAxes)
    # cax_dcl = inset_axes(ax["dcl"], **inset_kwargs, bbox_transform=ax["dcl"].transAxes)

    # Surface fields
    cbar_kwargs = dict(aspect=40, label="", orientation="horizontal", extend="both")
    surf_kwargs = dict(
        add_colorbar=True,
        x="time",
        robust=True,
        # ylim=[-5, 5],
        cbar_kwargs={
            "orientation": "horizontal",
            "label": "",
        },
    )

    hdl = subset.sst.plot(
        ax=ax["sst"],
        **surf_kwargs,
        cmap=mpl.cm.RdYlBu_r,
        # cbar_kwargs={"extend": "both", "label": ""},
    )
    #    f.colorbar(hdl, cax=cax_sst, **cbar_kwargs)

    hdl = (
        (subset.mld - subset.dcl_base)
        .resample(time="D")
        .mean()
        .plot(
            ax=ax["dcl"],
            **surf_kwargs,
            cmap=mpl.cm.GnBu,
            vmin=5,
            #  cbar_kwargs={"extend": "both", "label": ""},
        )
    )
    #   f.colorbar(hdl, cax=cax_dcl, **cbar_kwargs)

    subset.N2.where(subset.dcl_mask).mean("depth").plot(
        ax=ax["N2"], **cmaps["N2"], **surf_kwargs
    )

    shear_kwargs = dict(vmin=-0.02, vmax=0.02, x="time", cmap=mpl.cm.RdBu_r)

    subset.uz.sel(depth=slice(-60)).mean("depth").plot(
        ax=ax["uz"],
        **shear_kwargs,
        add_colorbar=False,
    )
    hdl = (
        subset.vz.sel(depth=slice(-60))
        .mean("depth")
        .plot(
            ax=ax["vz"],
            **shear_kwargs,
            add_colorbar=True,
            cbar_kwargs={"orientation": "horizontal", "label": "", "extend": "both"},
        )
    )

    fmt = mpl.ticker.ScalarFormatter()
    fmt.set_powerlimits((-1, 1))
    surf_kwargs["cbar_kwargs"]["format"] = fmt
    shred2 = (
        ((subset.S2 - subset.N2 / 0.4)).where(subset.dcl_mask).median("depth").compute()
    )
    shred2.plot(ax=ax["shred2"], **cmaps["shred2"], **surf_kwargs)

    sstmean = subset.sst.sel(latitude=slice(-3, None)).resample(time="D").mean()
    kwargs = dict(
        levels=[22.4, 23.75],
        x="time",
        add_labels=False,
    )
    for aa in axtop.flat:
        kwargs["ax"] = aa
        sstmean.plot.contour(colors="w", linewidths=1.5, **kwargs)
        sstmean.plot.contour(**kwargs, colors="k", linewidths=0.75)

    dcpy.plots.linex(times, color="k", zorder=2, ax=list(axtop.flat), lw=0.5)
    dcpy.plots.clean_axes(axtop)
    [aa.set_xlabel("") for aa in axtop[1, :]]
    [aa.set_title("") for aa in axtop.flat]
    for aa in axtop.flat:
        dcpy.plots.concise_date_formatter(aa, minticks=6, show_offset=False)
        [tt.set_rotation(0) for tt in aa.get_xticklabels()]

    dcpy.plots.label_subplots(
        axtop.flat,
        labels=[
            "SST",
            "$z_{MLD} - z_{Ri}$",
            "$u_z^2+v_z^2-N^2/Ri_c$",
            "$u_z$",
            "$v_z$",
            "$N^2$",
        ],
    )

    f.set_size_inches((dcpy.plots.pub_fig_width("jpo", "two column"), 6))


def plot_tiw_period_snapshots(full_subset, lon, period, times):
    subset = full_subset.sel(longitude=lon, method="nearest")
    subset.depth.attrs["units"] = "m"

    if period is not None:
        subset = subset.where(subset.period == period, drop=True)

    # times = subset.time.where(subset.tiw_phase.isin(np.arange(45, 290, 45)), drop=True)

    nextra = 2

    plt.rcParams["font.size"] = 9

    # right = gsparent[1].subgridspec(len(times), 4)
    # axx = np.empty((len(times), 4), dtype=np.object)
    # for icol in range(4):
    #     for irow in range(len(times)):
    #         axx[irow, icol] = f.add_subplot(right[irow, icol])

    f, axx = plt.subplots(3, 4, sharex=True, sharey=True, constrained_layout=True)

    ####### row snapshots
    sub = (
        subset[["u", "v", "uz", "vz", "N2", "Ri", "Jq"]]
        .sel(latitude=slice(-5, 5), depth=slice(-100))
        .sel(time=times, method="nearest")
        .compute()
    )

    for idx, (axrow, time) in enumerate(zip(axx, times)):
        tsub = sub.sel(time=time, method="nearest")

        add_colorbar = True if idx == (len(times) - 1) else False
        plot_shred2_time_instant(
            tsub, ax=dict(zip(["Jq", "Ri", "u", "v"], axrow)), add_colorbar=add_colorbar
        )
        # axrow[0].text(
        #     x=0.05,
        #     y=0.09,
        #     s=tsub.time.dt.strftime("%Y-%m-%d %H:%M").values,
        #     va="center",
        #     ha="left",
        #     transform=axrow[0].transAxes,
        # )
        if idx == len(times) - 1:
            [aa.tick_params(labelbottom=True) for aa in axrow]
        axrow[-1].text(
            x=1.03,
            y=0.5,
            s=tsub.time.dt.strftime("%d/%m %H:%M").values,
            va="center",
            rotation=270,
            transform=axrow[-1].transAxes,
        )
        if idx != (len(times) - 1):
            [aa.set_xlabel("") for aa in axrow]
        if idx != 0:
            [aa.set_title("") for aa in axrow]

    [aa.set_yticks([-100, -60, -30, 0]) for aa in axx.flat]
    [
        aa.set_yticklabels([str(num) for num in [-100, -60, -30, 0]])
        for aa in axx[:, 0].flat
    ]

    [
        aa.set_xticklabels(["", "4°S", "", "2°S", "", "0", "", "2°N", "", "4°N", ""])
        for aa in axx[-1, :]
    ]
    [aa.set_xlabel("") for aa in axx[-1, :]]
    dcpy.plots.label_subplots(axx.flat, y=0.85)

    return axx


def plot_dcl(subset, shear_max=False, zeros_flux=True, lw=2, kpp_terms=True):
    if "dens" not in subset:
        subset["dens"] = dens(subset.salt, subset.theta, np.array([0.0]))
        subset.dens.attrs["description"] = "Potential density, referenced to surface"
    if "S2" not in subset:
        subset = calc_reduced_shear(subset)
    if "mld" not in subset:
        subset["mld"] = get_mld(subset.dens)
    if "dcl_base" not in subset:
        subset["dcl_base"] = get_dcl_base_Ri(subset)

    varnames = [
        "KPPhbl",
        "nonlocal_flux",
        "mld",
        "dcl_base",
        "N2",
        "S2",
        "u",
        "Jq",
        "netflux",
        "stress",
        "Ri",
        # "KPPdiffKzT",
    ]
    if "KPPRi" in subset:
        varnames.append("KPPRi")

    if kpp_terms:
        varnames += [
            "taux",
            "tauy",
            "v",
            "theta",
            "salt",
            "short",
        ]
    subset = subset[varnames].compute()
    zeros = dcpy.util.interp_zero_crossing(subset.netflux)

    if kpp_terms:
        kpp = calc_kpp_terms(subset)
        # Rib = kpp.Rib

    if "KPPRi" in subset:
        Rib = subset["KPPRi"]
    else:
        if kpp_terms:
            Rib = kpp.Rib
        else:
            Rib = None

    stress = "stress" in subset
    if stress:
        subset.stress.attrs["long_name"] = "$τ$"
        subset.stress.attrs["units"] = "$N/m^2$"

    f, axx = plt.subplots(
        5,
        1,
        sharex=True,
        squeeze=False,
        constrained_layout=True,
        gridspec_kw=dict(height_ratios=[0.5, 1, 1, 1, 1]),
    )
    ax = axx.squeeze()
    ax[0].fill_between(x=subset.time.values, y1=subset.netflux, y2=0)
    ax[0].set_ylabel("$Q_{net}$ [$W/m²$]")

    if stress:
        axstress = ax[0].twinx()
        subset.stress.plot(ax=axstress, color="C1", ylim=(0, 0.1), x="time")
        axstress.set_title("")
        dcpy.plots.set_axes_color(axstress, "C1", spine="right")

        # (subset.Ri).plot(
        #     x="time",
        #     ax=ax[1],
        #     vmin=0.1,
        #     vmax=0.5,
        #     levels=np.arange(0.15, 0.85, 0.1),
        #     cmap=mpl.cm.RdGy_r,
        #     cbar_kwargs={"label": "$Ri$"},
        # )

        (subset.Ri).plot(x="time", ax=ax[1], **cmaps["Ri"])

    subset.S2.plot(x="time", ax=ax[2], **cmaps["S2"])
    subset.N2.plot(x="time", ax=ax[3], **cmaps["N2"])

    Jq = (
        (subset.Jq + subset.nonlocal_flux.fillna(0))
        .rolling(depth=2, center=True)
        .mean()
    )
    h = Jq.plot(
        x="time",
        ax=ax[4],
        **cmaps["Jq"],
        cbar_kwargs={"label": "$J_q^t$ [$W/m²$]"},
    )

    # subset.KPPdiffKzT.where(subset.KPPdiffKzT > 0).plot(
    #    ax=ax[4], norm=mpl.colors.LogNorm(), vmin=5e-4, vmax=5e-2, cmap=mpl.cm.YlGnBu_r,
    # )

    if shear_max:
        masked_S2 = subset.S2.where(
            (subset.depth > (subset.dcl_base.interpolate_na("time", "nearest") + 5))
            & (subset.u < 0.2)
        )
        masked_S2 = masked_S2.fillna(-1000)
        S2max = subset.depth[masked_S2.argmax("depth").compute()]

    locator = mpl.dates.AutoDateLocator()
    locator.intervald[mpl.dates.HOURLY] = [12]  # only show every 3 hours

    def plot(ax):

        if Rib is not None:
            hrib = (
                dcpy.interpolate.pchip_roots(
                    Rib.sortby("depth"),
                    "depth",
                    target=0.3,
                )
                .squeeze()
                .plot(ax=ax, _labels=False, lw=lw * 2.5, color="r")
            )

            dcpy.plots.annotate_end(hrib[0], "$Ri_b = 0.3$", va="top")
            # cs = Rib.plot.contour(
            #    ax=ax, levels=[0.3], linewidths=lw*2.5, colors="r", x="time"
            # )
        else:
            print("no Rib")
            cs = None

        hmld = subset.mld.plot(
            x="time",
            color="C1",
            ls="--",
            lw=lw + 1,
            ax=ax,
            label="$z_{MLD}$",
            _labels=False,
        )
        dcpy.plots.annotate_end(hmld[0], "$z_{MLD}$")

        (-1 * subset.KPPhbl).plot(x="time", color="w", lw=lw * 2, ax=ax, _labels=False)
        hkpp = (-1 * subset.KPPhbl).plot(
            x="time", color="k", lw=lw, ax=ax, label="$H_{KPP}$", _labels=False
        )
        dcpy.plots.annotate_end(hkpp[0], "$H_{KPP}$", va="bottom")

        subset.dcl_base.plot(color="k", lw=lw, ax=ax, _labels=False)
        (-1 * Jq).plot.contour(
            levels=17, colors="k", linewidths=0.5, x="time", robust=True, ax=ax
        )
        if zeros_flux:
            dcpy.plots.linex(zeros, ax=ax, zorder=3, color="w", lw=lw)

        if shear_max:
            S2max.plot(ax=ax)

        ax.xaxis.set_minor_locator(locator)

        if kpp_terms:
            (-1 * kpp.hmonob).plot(
                color="r",
                ls="none",
                ms=lw * 5,
                marker=".",
                label="$L_{MO}$",
                ax=ax,
                _labels=False,
            )

    [plot(axx) for axx in ax[1:]]
    # cs = plot(axx.flat[-1])
    # if cs:
    #    dcpy.plots.add_contour_legend(cs, "Ri_b", loc="lower right")
    dcpy.plots.clean_axes(axx)
    dcpy.plots.concise_date_formatter(axx.flat[-1], show_offset=False, minticks=5)
    # ax.flat[-1].legend(ncol=4)
    ax.flat[-1].set_xlabel("")
    f.set_size_inches((8, 10))

    return f, axx


def debug_kpp_plot(sub):
    """ Plot showing KPP terms estimated using a port of Bill Smyth's code. """

    h = calc_kpp_terms(sub)

    f, ax = plt.subplots(
        2,
        1,
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1, 2]},
        squeeze=False,
    )
    f.set_size_inches((8, 8))
    sub.netflux.plot(ax=ax[0, 0])

    plt.sca(ax[1, 0])
    cs = np.log10(sub.KPPdiffKzT.sel(depth=slice(-40))).plot.contour(
        levels=np.arange(-3, -1.5, 0.25),
        colors="k",
        robust=True,
        x="time",
        label="MIT $K_T$",
    )
    ax[1, 0].add_artist(
        ax[1, 0].legend(*cs.legend_elements("log_{10} K_T"), loc="lower right")
    )

    (sub.Jq.sel(depth=slice(-40))).plot(
        levels=21,
        robust=True,
        x="time",
    )

    def annotate(ax):
        cs = h.Rib.plot.contour(levels=[0.3], linewidths=3, colors="r", x="time")
        ax.add_artist(ax.legend(*cs.legend_elements("Ri_b")))
        # (-1 * h.hunlimit).plot(marker="o", markersize=8, label="unlimited hbl", ax=ax)
        (-1 * sub.KPPhbl).plot(lw=4, label="MIT hbl", ax=ax, color="k")
        (-1 * h.hmonob).plot(
            color="r", ls="--", marker="x", label="monin-obukhov", ax=ax
        )

    ax[-1, 0].legend()

    dcpy.plots.clean_axes(ax)


def plot_shear_terms(ds, duzdt, dvzdt, mask_dcl=True):
    def plot_dcl(dcl, ax):
        kwargs = dict(
            levels=[30],
            robust=True,
            x="time",
            add_labels=False,
            ax=ax,
        )

        dcl.isel(longitude=1).plot.contour(**kwargs, linewidths=3, colors="w")
        dcl.isel(longitude=1).plot.contour(**kwargs, linewidths=1.1, colors="k")

    def plot_shear_and_evol(shear, dcl, evol, fig, axes, colorbar=True, mask_dcl=True):
        hshear = shear.isel(longitude=1).plot(
            ax=axes["shear"],
            vmin=-0.02,
            vmax=0.02,
            x="time",
            add_colorbar=False,
            cmap=mpl.cm.RdBu_r,
        )

        mpl.rcParams["hatch.color"] = "lightgray"
        dclmask = xr.ones_like(dcl).where((dcl < 30))
        dclmask = xr.where(
            (dcl.latitude <= -5) & (dcl.latitude >= 4.8), 1.0, dclmask
        ).isel(longitude=1)

        for var in axes:
            if var not in evol:
                continue
            hevol = evol[var].plot(
                ax=axes[var],
                vmin=-5e-8,
                vmax=5e-8,
                cmap=mpl.cm.PuOr_r,
                x="time",
                add_colorbar=False,
            )

            if mask_dcl:
                dcpy.plots.plot_mask(axes[var], dclmask)
            else:
                plot_dcl(dcl, ax=axes[var])

        plot_dcl(dcl, ax=axes["shear"])
        return hshear, hevol

    def avg(ds):
        return ds.sel(depth=slice(-60), latitude=slice(-2, 6.5)).mean("depth")

    terms = ["shear",] + [
        term
        for term in ["xadv", "yadv", "str", "vtilt", "htilt", "buoy", "fric"]
        if term in duzdt
    ]
    with plt.rc_context({"font.size": 9}):

        f, axx = plt.subplots(
            len(terms), 2, sharex=True, sharey=True, constrained_layout=True
        )
        f.set_constrained_layout_pads(h_pad=0, w_pad=0)
        ax = dict()
        ax["u"] = dict(zip(terms, axx[:, 0]))
        ax["v"] = dict(zip(terms, axx[:, 1]))

        dcl = ds.dcl.resample(time="D").mean().rolling(latitude=3).mean().compute()

        hshear, hevol = plot_shear_and_evol(
            shear=avg(ds.uz),
            dcl=dcl,
            evol=avg(duzdt),
            fig=f,
            axes=ax["u"],
            colorbar=True,
            mask_dcl=mask_dcl,
        )

        plot_shear_and_evol(
            shear=avg(ds.vz),
            dcl=dcl,
            evol=avg(dvzdt),
            fig=f,
            axes=ax["v"],
            colorbar=False,
            mask_dcl=mask_dcl,
        )

        f.colorbar(
            hshear,
            ax=axx[0, :],
            # orientation="horizontal",
            extend="both",
            aspect=10,
            label="shear [$s^{-1}$]",
        )
        f.colorbar(
            hevol,
            ax=axx[1:, :],
            # orientation="horizontal",
            extend="both",
            aspect=5,
            shrink=0.3,
            label="shear tendency [$s^{-2}$]",
        )

        labelu = [
            "$u_z$",
        ] + [duzdt[var].term for var in terms[1:]]
        labelv = [
            "$v_z$",
        ] + [dvzdt[var].term for var in terms[1:]]
        labels = list(itertools.chain(*[(a, b) for a, b in zip(labelu, labelv)]))

        dcpy.plots.label_subplots(
            x=0.05,
            y=0.85,
            ax=axx.flat,
            labels=np.array(labels).flat,
            bbox=dict(color="white", alpha=0.9, pad=0.1),
        )
        dcpy.plots.clean_axes(axx)
        [aa.set_title("") for aa in axx[0, :]]
        [aa.set_xlabel("") for aa in axx[-1, :]]
        [aa.set_ylabel("") for aa in axx[:, 0]]
        [
            dcpy.plots.concise_date_formatter(aa, maxticks=6, show_offset=False)
            for aa in axx[-1, :]
        ]

        width = dcpy.plots.pub_fig_width("jpo", "medium 2")
        f.set_size_inches((width, width * 1.6))

    alltitles = dict(
        zip(
            ["xadv", "yadv", "str", "vtilt", "htilt", "buoy", "fric"],
            [
                "X ADV",
                "Y ADV",
                "STRETCH",
                "TILT1",
                "TILT2",
                "BUOY",
                "FRIC",
            ],
        )
    )
    titles = [alltitles[term] for term in terms[1:]]

    fontdict = {"fontsize": "small", "fontstyle": "oblique"}

    def right_label(ax, title, **kwargs):
        ax.annotate(
            title,
            xy=(1.02, 0.5),
            xycoords="axes fraction",
            rotation=270,
            ha="left",
            va="center",
            **kwargs,
        )

    [right_label(aa, tt, **fontdict) for aa, tt in zip(axx[1:, 1], titles)]

    return f, ax


def vor_streamplot(
    ds,
    vort,
    stride=8,
    refspeed=0.5,
    uy=True,
    vy=False,
    vec=True,
    stream=True,
    colorbar=True,
    x="time",
    longitude=-110,
    scale=0.6,
    ax=None,
):

    subset = vort.sel(latitude=slice(-2, 5), depth=slice(0, -60)).mean("depth")
    f0 = vort.f.reindex_like(subset.z)

    subset.z.attrs["long_name"] = "vert vorticity"

    if ax is None:
        f, ax = plt.subplots(1, 1, constrained_layout=True)
    else:
        f = ax.get_figure()

    cbar_kwargs = (
        {"label": "- $u_y$ [s$^{-1}$]", "orientation": "horizontal", "aspect": 25}
        if colorbar
        else None
    )
    if uy:
        ((-1 * subset.uy)).plot(
            y="latitude",
            vmax=2e-5,
            add_colorbar=colorbar,
            cbar_kwargs=cbar_kwargs,
            cmap=mpl.cm.PuOr_r,
            ax=ax,
        )

    if vy:
        ((subset.vy)).plot(
            y="latitude",
            vmax=1e-5,
            add_colorbar=colorbar,
            cbar_kwargs=cbar_kwargs,
            cmap=mpl.cm.PuOr_r,
            ax=ax,
        )

    # dataset at central longitude
    if x == "time":
        center = ds.sel(longitude=longitude, method="nearest")
    else:
        center = ds

    if "dcl" in center:
        dcl = center.dcl
        if x == "time":
            dcl = dcl.resample(time="D", loffset="12H").mean()
        elif x == "longitude":
            dcl = dcl.rolling(longitude=6, center=True).mean()

        dcl.load()
        kwargs = dict(ax=ax, y="latitude", zorder=2, add_labels=False, levels=[30])
        dcl.plot.contour(**kwargs, colors="w", linewidths=4)
        dcl.plot.contour(**kwargs, colors="k")

    # cs = (
    #    ds.theta.isel(depth=0, longitude=1)
    #    .resample(time="D", loffset="12H")
    #    .mean()
    #    .plot.contour(x="time", colors="k", levels=[23], zorder=2, add_labels=False)
    # )
    # dcpy.plots.contour_label_spines(cs)

    if vec:
        quiver(
            subset[["x", "y"]]
            .isel({x: slice(None, None, 6)})
            .isel({"latitude": slice(None, None, stride), x: slice(None, None, stride)})
            .compute(),
            u="x",  # vorticity x-component
            v="y",  # vorticity y-component
            x=x,
            y="latitude",
            scale=scale,
            ax=ax,
        )

    subset = (
        center.sel(latitude=slice(-1, 5), depth=slice(-60))
        .isel(latitude=slice(0, -1, 1))
        .mean("depth")
    )

    if x == "time":
        dx = (
            (subset.time.dt.round("H") - subset.time[0])
            .values.astype("timedelta64[s]")
            .astype("int")
        ) / 120e3
    elif x == "longitude":
        dx = np.arange(0, 0.05 * subset.sizes["longitude"], 0.05)
        # dx = (subset[x] - subset[x][0]).round(2).values
        # tiny error accumulating resulting in uneven grid spacing
        # streamplot throws error if we don't make this correction
        # dx[dx > 7.05] -= 0.01

        # dx *= 110e3
        # dx += subset[x][0].values

    if stream:
        sax = ax.twiny()

        dx, y, u, v = dask.compute(
            dx,
            subset.latitude,
            subset.u.transpose("latitude", x) + refspeed,
            subset.v.transpose("latitude", x),
        )

        sax.streamplot(
            dx,
            y.values,
            u.values,
            v.values,
            color="w",
            linewidth=3,
            arrowstyle="-",
            density=0.8,
        )
        sax.streamplot(
            dx,
            y.values,
            u.values,
            v.values,
            color="seagreen",
            linewidth=1.4,
            density=0.8,
        )
        sax.set_xticks([])
        sax.set_xticklabels([])

    euc = center.u.sel(latitude=slice(-2, 4)).sel(depth=slice(-30, -120)).mean("depth")
    euclat = euc.latitude[euc.argmax("latitude")].compute()

    euclat.plot(ax=ax, lw=6, color="white", zorder=10)
    heuc = euclat.plot(ax=ax, lw=3, color="dimgray", zorder=11)
    dcpy.plots.annotate_end(heuc[0], "EUC core")

    # f.suptitle(
    #    "Horizontal vorticity vectors [black]; relative velocity streamlines [green].\n Depth average to 60m",
    #    y=1.05,
    # )
    ax.set_title("")
    # ax.set_ylim([-1, 5])
    ax.set_ylabel("Latitude [°N]")
    if x == "time":
        ax.set_xlabel("")
        dcpy.plots.concise_date_formatter(ax, show_offset=False)
    elif x == "longitude":
        ax.set_xlabel("Longitude [°E]")

    # TODO: dcpy.plots.contour_label_spines(cs, "SST=")

    if colorbar:
        f.set_size_inches((dcpy.plots.pub_fig_width("jpo", "medium 1"), 5))
    else:
        f.set_size_inches((5.3, 4.3))


def plot_reference_speed(ax):

    ax.axvline(-110)

    speeds = [
        -0.3,
        -0.4,
        -0.5,
        -0.6,
    ]  # m/s

    tlim = pd.to_datetime(mpl.dates.num2date(ax.get_ylim()))
    x0 = -110
    t0 = tlim[0] + pd.Timedelta("5D")
    for spd in speeds:
        x1 = x0 + spd / 1000 / 110 * 15 * 86400
        ax.plot([x0, x1], [t0, t0 + pd.Timedelta("15D")], zorder=10, color="w", lw=4)
        ax.plot([x0, x1], [t0, t0 + pd.Timedelta("15D")], zorder=10, color="k", lw=2)


plot_reference = plot_reference_speed


def plot_daily_cycles(ds):

    f, axx = plt.subplots(4, 1, sharex=True, sharey=True, constrained_layout=True)
    ax = dict(zip(["Ri", "S2", "N2", "Jq"], axx))

    assert all([key in ds for key in ax.keys()])

    for var in ax.keys():
        ds[var].plot(ax=ax[var], x="time", y="depth", **cmaps[var])

    def plot(ax):
        ds.mld.plot(ax=ax, color="w", x="time", _labels=False, lw=2)
        hmld = ds.mld.plot(ax=ax, color="darkorange", x="time", _labels=False, lw=1)
        ds.dcl_base.plot(ax=ax, color="w", x="time", _labels=False, lw=2)
        hdcl = ds.dcl_base.plot(ax=ax, color="k", x="time", _labels=False, lw=1)
        # (-1 * ds.KPPhbl).plot(ax=ax, color='r', x="time", _labels=False)
        dcpy.plots.annotate_end(hmld[0], "$z_{MLD}$", va="bottom")
        dcpy.plots.annotate_end(hdcl[0], "$z_{Ri}$", va="top")
        return hmld, hdcl

    [plot(aa) for aa in axx]
    # hmld, hdcl = plot(axx[-1])

    [aa.set_xlabel("") for aa in axx]
    [aa.set_title("") for aa in axx[1:]]
    [tt.set_rotation(0) for tt in axx[-1].get_xticklabels()]
    dcpy.plots.concise_date_formatter(axx[-1], minticks=9, show_offset=False)
    axx[0].set_title("3.5°N, 110°W")
    dcpy.plots.label_subplots(axx, y=0.05, bbox=dict(color="white", alpha=0.8, pad=0.1))

    f.set_size_inches((dcpy.plots.pub_fig_width("jpo", "medium 2"), 6))
    return f, ax
