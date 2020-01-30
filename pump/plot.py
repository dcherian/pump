import dcpy.plots
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from .calc import get_dcl_base_Ri, get_mld


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


def plot_jq_sst(model, lon, periods, lat=0):

    sst = model.surface.theta.sel(longitude=lon, method="nearest")

    period = model.full.period.sel(longitude=lon, method="nearest")
    tperiod = sst.time.where(period.isin(periods), drop=True)[[0, -1]]
    tperiod = slice(*list(tperiod.values))
    tao = model.tao.sel(longitude=lon, time=tperiod, depth=slice(0, -500))

    lat = np.atleast_1d(lat)

    region = dict(latitude=lat, method="nearest")

    mld = get_mld(tao.dens.sel(**region)).compute()
    dcl_base = get_dcl_base_Ri(tao.sel(**region), mld).compute()
    # tao = model.full.sel(longitude=lon, time=tperiod, depth=slice(0, -500))

    f, ax = plt.subplots(
        2 + len(lat),
        1,
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2] * (1 + len(lat)) + [1]},
    )

    # First SST
    plt.sca(ax[0])
    sst.sel(time=tperiod).plot(
        x="time", cmap=mpl.cm.RdYlBu_r, robust=True, ylim=[-8, 8]
    )
    dcpy.plots.liney(
        tao.sel(**region).latitude, ax=ax[0], ls="--", color="k", lw=1, zorder=10
    )

    # Jq qwith eucmax, MLD, DCL
    for la, axis in zip(lat[::-1], ax[1:-1]):
        region = {"latitude": la, "method": "nearest"}
        plt.sca(axis)
        tao.sel(**region).Jq.rolling(depth=3).mean().plot(
            x="time", vmax=0, robust=True, cmap=mpl.cm.GnBu
        )
        dcl_base.sel(**region).plot(x="time", color="k", _labels=False)
        mld.sel(**region).plot(x="time", color="r", _labels=False)
        tao.sel(latitude=0).euc_max.plot(x="time", color="w", _labels=False)
        axis.set_ylim([-150, 0])

    # Just tiw phase
    plt.sca(ax[-1])
    model.full.tiw_phase.sel(longitude=lon, method="nearest").plot(_labels=False)
    [aa.set_xlabel("") for aa in ax[:-1]]


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
