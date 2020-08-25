import colorcet
import dask
import dcpy.plots
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from .calc import get_dcl_base_Ri, get_mld
import xarray


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

    kwargs.pop("cmap_params")
    kwargs.pop("hue")
    kwargs.pop("hue_style")
    hdl = ax.quiver(x.values, y.values, u.values, v.values, scale=scale, **kwargs)

    return hdl


def plot_depths(ds, ax=None, **kwargs):

    if ax is None:
        ax = plt.gca()

    if "euc_max" in ds:
        heuc = ds.euc_max.plot.line(ax=ax, color="k", lw=1, _labels=False, **kwargs)

    if "dcl_base" in ds:
        hdcl = ds.dcl_base.plot.line(ax=ax, color="gray", lw=1, _labels=False, **kwargs)

    if "mld" in ds:
        hmld = (ds.mld).plot.line(ax=ax, color="k", lw=0.5, _labels=False, **kwargs)


def plot_bulk_Ri_diagnosis(ds, f=None, ax=None, **kwargs):
    """
    Estimates fractional contributions of various terms to bulk Richardson
    number.
    """

    def plot_ri_contrib(ax1, ax2, v, factor=1, **kwargs):
        # Better to call differentiate on log-transformed variable
        # This is a nicer estimate of the gradient and is analytically equal
        per = factor * np.log(np.abs(v)).compute().differentiate("longitude")
        hdl = per.plot(
            ax=ax2,
            x="longitude",
            label=f"{factor}/{v.name} $∂_x${v.name}",
            add_legend=False,
            **kwargs,
        )
        v.plot(ax=ax1, x="longitude", **kwargs)
        ax1.set_xlabel("")
        ax1.set_title("")

        return per, hdl

    if f is None and ax is None:
        f, axx = plt.subplots(
            7,
            1,
            constrained_layout=True,
            sharex=True,
            gridspec_kw={"height_ratios": [1, 1, 1, 1, 1, 1, 2]},
        )
        ax = dict(zip(["Rib", "h", "du", "db", "u", "b", "contrib"], axx))
        add_legend = True
    else:
        add_legend = False

    colors = dict(
        {
            "us": "C0",
            "ueuc": "C1",
            "bs": "C0",
            "beuc": "C1",
            "Rib": "C0",
            "h": "C1",
            "du": "C2",
            "db": "C3",
        }
    )

    factor = dict(zip(ax.keys(), [1, 1, -2, 1]))
    rhs = xr.zeros_like(ds.bs)
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

    for vv in ["u", "b"]:
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
    ax["b"].set_ylim([-0.0007, 0.0007])

    ax["du"].set_ylim([-1.3, -0.3])
    ax["db"].set_ylim([0.005, 0.05])

    ax["Rib"].set_ylabel("Ri$_b =  Δbh/Δu²$")
    ax["Rib"].set_yscale("log")
    ax["Rib"].set_yticks([0.25, 0.5, 1, 5, 10])
    ax["Rib"].grid(True)

    rhs.plot(
        ax=ax["contrib"],
        x="longitude",
        color="k",
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
        eucmax = eucmax.sel(longitude=lon, method="nearest").sel(time=tperiod)

    lat = np.atleast_1d(lat)

    region = dict(latitude=lat, method="nearest")

    mld = get_mld(tao.dens.sel(**region))
    dcl_base = get_dcl_base_Ri(tao.sel(**region), mld, eucmax)

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
    f.set_constrained_layout_pads(wspace=0, w_pad=0)

    ax[0, 1].remove()
    ax[1, 1].remove()

    # First SST
    plt.sca(ax[0, 0])
    # cax = dcpy.plots.cbar_inset_axes(ax[0, 0])
    sst.name = "SST"
    sst.attrs["units"] = "°C"
    sst.sel(time=tperiod, latitude=slice(-2, 5)).plot(
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
        levels=np.sort([-600, -450, -300, -200, -150, -100, -75, -50, -25, 0]),
        cmap=mpl.cm.Blues_r,
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
                # "orientation": "horizontal",
                #    "shrink": 0.8,
                #    "aspect": 40,
                "label": "",
                "ticks": [-600, -400, -200, -100, -50, 0],
                # "cax": cax,
            },
        )
    )
    dcpy.plots.liney(
        tao.sel(**region).latitude, ax=ax[0, 0], ls="--", color="k", lw=1, zorder=10
    )
    ax[1, 0].set_xlabel("")
    ax[1, 0].set_title("")

    dcpy.plots.liney(
        tao.sel(**region).latitude, ax=ax[:2, 0], ls="--", color="k", lw=1, zorder=10
    )

    # Jq with eucmax, MLD, DCL
    for index, (la, axis, axRi) in enumerate(zip(lat[::-1], ax[2:, 0], ax[2:, 1])):
        tRi = time_Ri[la]
        region = {"latitude": la, "method": "nearest"}
        plt.sca(axis)
        jq = tao.sel(**region).Jq.rolling(depth=2, min_periods=1, center=True).mean()
        jq.plot.contourf(**jq_kwargs, add_colorbar=False)

        rii = (
            tao.Ri.sel(time=tRi)
            .where((tao.depth < mld) & (tao.depth > dcl_base))
            .sel(**region)
            .chunk({"time": -1})
        )
        rii = rii.where(rii.count("time") > 15)
        ri_q = (rii.chunk({"time": -1}).quantile(dim="time", q=[0.25, 0.5, 0.75])).compute()

        # .plot.line(ax=axRi, xscale="log", hue="quantile", y="depth", xlim=(0.1, 2))

        # mark Ri distribution time
        t = pd.date_range(tRi.start, tRi.stop)
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
        ri_q.sel(quantile=0.5).plot.line(ax=axRi, xscale="log", y="depth", color="r")
        axRi.set_xlim((0.1, 2))
        dcpy.plots.linex([0.25, 0.4], ax=axRi)
        axRi.set_title("")
        axRi.set_ylabel("")

        hdl = dcl_base.sel(**region).plot(x="time", color="k", _labels=False)
        dcpy.plots.annotate_end(hdl[0], "$z_{Ri}$", va="top")
        hdl = mld.sel(**region).plot(x="time", color="C1", _labels=False)
        dcpy.plots.annotate_end(hdl[0], "$z_{MLD}$", va="bottom")
        if eucmax is not None and np.abs(la) < 2:
            hdl = eucmax.plot(x="time", color="k", linestyle="--", _labels=False)
            dcpy.plots.annotate_end(hdl[0], "$z_{EUC}$")

        axis.set_ylim([-100, 0])
        axis.set_title(f"latitude={la}°N")
        axis.set_xlabel("")
        # axis.text(0.03, 0.05, f"latitude={la}°N", color="k", transform=axis.transAxes)

    ax[1, 1].set_xlabel("")
    # Just tiw phase
    # plt.sca(ax[-1])
    # model.full.tiw_phase.sel(longitude=lon, method="nearest").plot(_labels=False)
    # [aa.set_xlabel("") for aa in ax[:-1]]

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


def plot_shear_terms(shear, dcl=None):
    kwargs = dict(
        col="term",
        x="time",
        robust=True,
        cbar_kwargs={
            "orientation": "horizontal",
            "shrink": 0.6,
            "aspect": 40,
            "pad": 0.15,
        },
        vmin=-5e-8,
        vmax=5e-8,
        cmap=mpl.cm.RdBu_r,
    )
    if "depth" in shear.dims:
        kwargs["row"] = "depth"

    fg = shear.sel(latitude=slice(-3, 5)).to_array("term").plot(size=5, **kwargs)
    if "name" in shear.attrs:
        fg.fig.suptitle(shear.attrs["name"], y=1.01)

    def plot():
        dcl.plot.contour(
            levels=7, colors="k", robust=True, x="time", add_labels=False, linewidths=1
        )

    if dcl is not None:
        fg.map(plot)


def plot_shred2_time_instant(tsub, ax, add_colorbar):

    kwargs = dict(
        # vmin=-0.02,
        # vmax=0.02,
        norm=mpl.colors.TwoSlopeNorm(vcenter=-5e-7, vmin=-5e-4, vmax=1e-4),
        cmap=mpl.cm.RdBu_r,
        add_colorbar=False,
    )

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
        cbar_kwargs={"orientation": "horizontal", "ax": [ax["Jq"]]}
        if add_colorbar
        else None,
    )

    def annotate(tsub, ax):
        tsub.dcl_base.plot(ax=ax, color="w", lw=2, _labels=False)
        tsub.dcl_base.plot(ax=ax, color="k", lw=1, _labels=False)
        tsub.mld.plot(ax=ax, color="w", lw=2, _labels=False)
        tsub.mld.plot(ax=ax, color="orange", lw=1, _labels=False)
        # dcpy.plots.liney(tsub.eucmax, ax=ax, zorder=10, color="k")

    axx = list(ax.values())
    mask = xr.where((tsub.depth > tsub.mld) | (tsub.depth < tsub.dcl_base), 1, np.nan)
    [dcpy.plots.plot_mask(aa, mask) for aa in axx]

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
    [aa.tick_params(top=False,) for aa in axx]
    [aa.tick_params(labelbottom=False) for aa in axx]
    [annotate(tsub, aa) for aa in axx]


def plot_tiw_period_snapshots(full_subset, lon, period, times):
    subset = full_subset.sel(longitude=lon, method="nearest")
    subset.depth.attrs["units"] = "m"

    if period is not None:
        subset = subset.where(subset.period == period, drop=True)

    # times = subset.time.where(subset.tiw_phase.isin(np.arange(45, 290, 45)), drop=True)

    nextra = 2

    plt.rcParams["font.size"] = 9

    f = plt.figure(constrained_layout=True)
    width = dcpy.plots.pub_fig_width("jpo", "two column")
    f.set_size_inches((width, 8.5))
    gsparent = f.add_gridspec(2, 1, height_ratios=[1, 1.5])
    left = gsparent[0].subgridspec(2, 2)
    ax = dict()
    ax["sst"] = f.add_subplot(left[0, 0])
    ax["dcl"] = f.add_subplot(left[0, 1], sharex=ax["sst"], sharey=ax["sst"])
    ax["uz"] = f.add_subplot(left[1, 0], sharex=ax["sst"], sharey=ax["sst"])
    ax["vz"] = f.add_subplot(left[1, 1], sharex=ax["sst"], sharey=ax["sst"])

    axtop = np.array(list(ax.values())).reshape((2, 2))

    # cax_sst = inset_axes(ax["sst"], **inset_kwargs, bbox_transform=ax["sst"].transAxes)
    # cax_dcl = inset_axes(ax["dcl"], **inset_kwargs, bbox_transform=ax["dcl"].transAxes)

    right = gsparent[1].subgridspec(len(times), 4)
    axx = np.empty((len(times), 4), dtype=np.object)
    for icol in range(4):
        for irow in range(len(times)):
            axx[irow, icol] = f.add_subplot(right[irow, icol])

    # Surface fields
    cbar_kwargs = dict(aspect=40, label="", orientation="horizontal", extend="both")
    surf_kwargs = dict(add_colorbar=True, x="time", robust=True, ylim=[-5, 5],)

    hdl = subset.sst.plot(
        ax=ax["sst"],
        **surf_kwargs,
        cmap=mpl.cm.RdYlBu_r,
        cbar_kwargs={"extend": "both", "label": ""},
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
            cbar_kwargs={"extend": "both", "label": ""},
        )
    )
    #   f.colorbar(hdl, cax=cax_dcl, **cbar_kwargs)

    shear_kwargs = dict(vmin=-0.02, vmax=0.02, x="time", cmap=mpl.cm.RdBu_r)

    subset.uz.sel(depth=slice(-60)).mean("depth").plot(
        ax=ax["uz"], **shear_kwargs, add_colorbar=False,
    )
    hdl = (
        subset.vz.sel(depth=slice(-60))
        .mean("depth")
        .plot(
            ax=ax["vz"],
            **shear_kwargs,
            add_colorbar=True,
            cbar_kwargs={"extend": "both", "label": ""},
        )
    )
    # f.colorbar(hdl, ax=axtop.flat[-1:], extend="both")

    dcpy.plots.linex(times, color="k", zorder=2, ax=list(axtop.flat), lw=0.5)
    dcpy.plots.clean_axes(axtop)
    [aa.set_xlabel("") for aa in axtop[1, :]]
    [aa.set_title("") for aa in axtop.flat]
    for aa in axtop.flat:
        dcpy.plots.concise_date_formatter(aa, minticks=6, show_offset=False)

    dcpy.plots.label_subplots(
        axtop.flat, labels=["SST", "$z_{MLD} - z_{Ri}$", "$u_z$", "$v_z$"]
    )

    sstmean = subset.sst.sel(latitude=slice(-3, None)).resample(time="D").mean()
    kwargs = dict(levels=[22.4, 23.75], x="time", add_labels=False,)
    for aa in axtop.flat:
        kwargs["ax"] = aa
        sstmean.plot.contour(colors="w", linewidths=1.5, **kwargs)
        sstmean.plot.contour(**kwargs, colors="k", linewidths=0.75)

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
        axrow[0].text(
            x=0.05,
            y=0.05,
            s=tsub.time.dt.strftime("%Y-%m-%d %H:%M").values,
            va="center",
            ha="left",
            transform=axrow[0].transAxes,
        )
        if idx == len(times) - 1:
            [aa.tick_params(labelbottom=True) for aa in axrow]
        # axrow[-1].text(
        #    x=1.05,
        #    y=0.5,
        #    s=tsub.time.dt.strftime("%Y-%m-%d %H").values,
        #    va="center",
        #    rotation=90,
        #    transform=axrow[-1].transAxes,
        # )
        if idx != (len(times) - 1):
            [aa.set_xlabel("") for aa in axrow]
        if idx != 0:
            [aa.set_title("") for aa in axrow]

    # ax["sst"].set_xticklabels([])
    # ax["sst"].set_xlabel("")
    # ax["sst"].set_title("SST [°C]")
    # ax["dcl"].set_title("Low Ri layer width [m]")
    # ax["dcl"].set_xlabel("")
    # dcpy.plots.concise_date_formatter(ax["sst"], minticks=6)

    [aa.set_yticks([-100, -60, -30, 0]) for aa in axx.flat]
    [aa.set_yticklabels([str(num) for num in [-100, -60, -30, 0]]) for aa in axx[:, 0]]
    [aa.set_yticklabels([]) for aa in axx[:, 1:].flat]
    # [tt.set_visible(True) for tt in aa.get_yticklabels() for aa in axx[:, 0]]

    [
        aa.set_xticklabels(["", "4°S", "", "2°S", "", "0", "", "2°N", "", "4°N", ""])
        for aa in axx[-1, :]
    ]
    [aa.set_xlabel("") for aa in axx[-1, :]]
    dcpy.plots.label_subplots(axx.flat, start="e")

    return axx
