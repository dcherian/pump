import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tqdm
import xarray as xr

import dcpy


def read_adcp(filename, longitude, debug=False):
    adcp = xr.load_dataset(filename)

    if "lat" in adcp:
        adcp = adcp.rename({"lat": "latitude", "lon": "longitude"})

    if "depth_cell" in adcp.dims:
        # xr.testing.assert_allclose(
        #    adcp.depth.diff("time"), xr.zeros_like(adcp.depth.diff("time"))
        # )
        adcp["depth_cell"] = adcp.depth.isel(time=0)
        adcp = adcp.drop_vars("depth").rename({"depth_cell": "depth"})

    adcp = adcp.set_coords(["latitude", "longitude"])
    section_mask = (
        (adcp.longitude <= longitude + 1)
        & (adcp.longitude >= longitude - 1)
        & (np.abs(adcp.latitude) < 8)
    )

    adcp = (
        adcp.interpolate_na("depth")
        .interpolate_na("time")
        .where(section_mask, drop=True)
    )

    if debug:
        adcp.plot.scatter("longitude", "latitude", s=2)

    return adcp


def trim_ctd_adcp(ctd, adcp):
    import dask

    start_time = np.max([adcp.time[0].values, ctd.time[0].values])
    stop_time = np.min([adcp.time[-1].values, ctd.time[-1].values])
    time_slice = slice(start_time, stop_time)

    ctd = ctd.swap_dims({"latitude": "time"}).sel(time=time_slice)
    if "time" not in adcp.dims:
        adcp = adcp.swap_dims({"latitude": "time"})
    adcp = adcp.sel(time=time_slice).sel(time=ctd.time.values, method="nearest")
    adcp = adcp.swap_dims({"time": "latitude"})
    ctd = ctd.swap_dims({"time": "latitude"})

    return dask.compute(ctd, adcp)


def find_cruise(cruises, code):
    for idx, cruise in enumerate(cruises):
        if code in cruise.attrs["EXPOCODE"]:
            print(f"Found {code} at index {idx}.")
            cruise["density"] = dcpy.eos.pden(
                cruise.salinity, cruise.temperature, cruise.pressure
            )
            return cruise


def preprocess(ds):
    return ds.squeeze().expand_dims(["time"]).assign(longitude=-110)


def read_cruise_folders(dirname):

    cruises = []
    dirs = glob.glob(dirname)
    for folder in tqdm.tqdm(dirs):
        ds = xr.open_mfdataset(
            f"{folder}/*.nc",
            combine="nested",
            parallel=True,
            preprocess=preprocess,
            concat_dim="time",
        )
        # ds["latitude"] = (("time",), np.atleast_1d(ds.latitude.values))
        if ds.sizes["time"] == 1:
            continue
        ds = ds.swap_dims({"time": "latitude"})
        ds["density"] = dcpy.eos.pden(ds.salinity, ds.temperature, ds.pressure)
        drho = ds.density - ds.density.bfill("pressure").isel(pressure=0)
        ds["mld"] = xr.where(drho > 0.015, drho.pressure, np.nan).min("pressure")

        cruises.append(ds)

    return cruises


def get_nice_cols(nfacets):
    rem = []
    ncol = np.array([2, 3, 4])
    for nn in ncol:
        rem.append(nfacets % nn)
    rem = np.array(rem)

    if np.any(rem == 0):
        nc = ncol[rem == 0].max()
    else:
        nc = ncol[rem.argmax()]

    return nc


def plot_ctd_stations_sst(cruises, oisst):
    import os

    for cruise in tqdm.tqdm(cruises):  # cruise = cruises[11]
        source = os.path.split(cruise.encoding["source"])[-1][:-3]
        filename = f"../images/ctds/{source}.png"

        cruise = cruise.where(np.abs(cruise.latitude) < 8, drop=True)

        trange = slice(*cruise.time.values[[0, -1]])

        subset = oisst.sel(time=trange)
        if subset.sizes["time"] == 0:
            print(f"skipping {source}: {trange}")
            continue

        fg = subset.plot(
            col="time",
            col_wrap=get_nice_cols(subset.sizes["time"]),
            robust=True,
            cmap=mpl.cm.RdYlBu_r,
            cbar_kwargs={"orientation": "horizontal", "shrink": 0.8, "aspect": 40},
        )

        for loc, ax in zip(fg.name_dicts.flat, fg.axes.flat):
            if loc is None:
                continue
            timestr = pd.Timestamp(list(loc.values())[0]).strftime("%Y-%m-%d")
            ax.plot(
                [-110] * cruise.sizes["latitude"],
                cruise.latitude,
                marker="o",
                color="k",
                markersize=4,
            )
            lats = np.atleast_1d(
                cruise.swap_dims({"latitude": "time"}).sel(time=timestr).latitude
            )
            ax.plot(
                [-110] * len(lats), lats, marker="o", ls="none", color="w", markersize=2
            )
            dcpy.plots.liney([0, 1, 2, 3, 4], ax=ax, lw=1, color="w", zorder=4)

        fg.fig.suptitle(
            f"{source} | EXPOCODE: {cruise.attrs['EXPOCODE']}", x=0.4, y=1.04
        )
        fg.fig.savefig(filename, bbox_inches="tight", dpi=180)
        plt.close(fg.fig)


def grid_ctd_adcp(ctd, adcp):

    bins = get_bins_around_levels(adcp.depth.values)
    binned = (
        ctd.groupby_bins("pressure", bins).mean().rename({"pressure_bins": "depth"})
    )
    new_depths = [interval.mid for interval in binned.depth.values]
    binned["depth"] = new_depths

    lats = binned.latitude.reset_coords(drop=True)

    binned["uz"] = adcp.u.differentiate("depth").assign_coords(latitude=lats)
    binned["vz"] = adcp.v.differentiate("depth").assign_coords(latitude=lats)
    binned["S2"] = binned.uz ** 2 + binned.vz ** 2
    binned["N2"] = 9.81 / 1025 * binned.density.compute().differentiate("depth")
    binned["Ri"] = binned.N2.where(binned.N2 > 1e-6) / binned.S2.where(binned.S2 > 1e-8)
    binned["mld"] = ctd.mld

    return binned


def get_bins_around_levels(levels):
    dz = np.diff(levels)
    return np.append(
        np.insert((levels[:-1] + levels[1:]) / 2, 0, levels[0] - dz[0] / 2),
        levels[-1] + dz[-1] / 2,
    )


def plot_section(ctd, adcp, binned, oisst, ladcp=None):

    expected_lat = [0, 0.5, 1, 1.5, 2, 3, 3.5, 4, 4.5, 5]

    f = plt.figure(constrained_layout=True)
    gsparent = f.add_gridspec(2, 1, height_ratios=[1.5, 1])
    ax = dict()

    gs0 = gsparent[0].subgridspec(2, 2)
    ax["sst"] = f.add_subplot(gs0[0, 0])
    ax["Ri"] = f.add_subplot(gs0[0, 1])
    # ax["dens"] = f.add_subplot(gs[3], sharex=ax["Ri"], sharey=ax["Ri"])

    # gs1 = gsparent[1].subgridspec(1, 2)
    ax["u"] = f.add_subplot(gs0[1, 0], sharex=ax["Ri"])
    ax["v"] = f.add_subplot(gs0[1, 1], sharex=ax["Ri"])

    gs2 = gsparent[1].subgridspec(1, len(expected_lat))
    # ax["lats"] = [f.add_subplot(gs2[0])]
    ax["lats"] = [f.add_subplot(gs2[nn]) for nn in range(len(expected_lat))]

    # SST
    obs_days = np.unique(ctd.time.mean("latitude").dt.round("D"))
    (
        oisst.sel(time=obs_days, method="nearest")
        .sel(lon=slice(-140, None), lat=slice(-3, 7))
        .mean("time")
        .plot(
            ax=ax["sst"],
            robust=True,
            cmap=mpl.cm.RdYlBu_r,
            vmin=22,
            vmax=26,
            # cbar_kwargs={"orientation": "horizontal"}
        )
    )
    ax["sst"].plot(
        ctd.longitude.broadcast_like(ctd.latitude),
        ctd.latitude,
        marker="o",
        color="k",
        markersize=4,
    )

    # Ri with u,v,ρ
    # for axx in [ax["u"], ax["v"], ax["Ri"]]:
    #     hdl = binned.Ri.plot(
    #         ax=axx,
    #         y="depth",
    #         yincrease=False,
    #         norm=mpl.colors.LogNorm(0.25, 1),
    #         cmap=mpl.cm.Blues,
    #         add_colorbar=False,
    #     )

    Ric = 0.27
    norm = mpl.colors.DivergingNorm(vcenter=-1e-7, vmin=-5e-3, vmax=1e-4)

    shred2 = binned.S2 - 1 / Ric * binned.N2
    ushred = binned.uz ** 2 - 1 / Ric / 2 * binned.N2
    vshred = binned.vz ** 2 - 1 / Ric / 2 * binned.N2
    shred2_kwargs = dict(
        y="depth",
        yincrease=False,
        cmap=mpl.cm.RdBu_r,
        add_colorbar=True,
        norm=norm,
    )
    hdl = shred2.plot(ax=ax["Ri"], **shred2_kwargs)
    ushred.plot(ax=ax["u"], **shred2_kwargs)
    vshred.plot(ax=ax["v"], **shred2_kwargs)

    color = "gray"
    adcp.u.plot.contour(ax=ax["u"], y="depth", colors=color, levels=10, ylim=(150, 0))
    adcp.v.plot.contour(ax=ax["v"], y="depth", colors=color, levels=10, ylim=(150, 0))
    ctd.density.plot.contour(
        ax=ax["Ri"], y="pressure", colors=color, levels=15, ylim=(150, 0)
    )
    # f.colorbar(hdl, ax=[ax["u"], ax["v"], ax["Ri"]])

    for axx in [ax["u"], ax["v"], ax["Ri"]]:
        binned.mld.plot(ax=axx, color="k", lw=2)

    plotted_lats = []
    # Ri profiles
    for axes, lat, expect_lat in zip(
        ax["lats"],
        binned.latitude.sel(latitude=expected_lat, method="nearest").values,
        expected_lat,
    ):

        # print([lat, expect_lat])
        # print(plotted_lats)
        # import IPython; IPython.core.debugger.set_trace()
        if (
            np.abs(lat - expect_lat) > 0.3
            # and np.round(lat, 1) != 3.5
            and np.round(lat, 2) not in plotted_lats
        ):
            print(f"skipping {expect_lat}")
            axes.remove()
            continue

        plotted_lats.append(lat)

        binned.Ri.sel(latitude=lat).plot(
            ax=axes,
            y="depth",
            xscale="log",
            yincrease=False,
            xlim=(0.1, 4),
            ylim=(100, 0),
            marker=".",
            color="k",
            _labels=False,
        )

        axrho = axes.twiny()

        rem = ctd.sizes["pressure"] % 3
        if rem != 0:
            dens = ctd.density.isel(pressure=slice(-rem))
        else:
            dens = ctd.density
        drho = (
            9.81 / 1025 * dens.coarsen(pressure=3).mean().differentiate("pressure")
        )  # ctd.density - ctd.density.bfill("pressure").isel(pressure=0)
        (drho).sel(latitude=lat, pressure=slice(0, 150)).plot(
            ax=axrho,
            y="pressure",
            ylim=(100, 0),
            _labels=False,
            xlim=(1e-6, 5e-4),
            xscale="log",
        )
        axrho.set_xticks([1e-6, 1e-5, 1e-4])
        dcpy.plots.liney(binned.mld.sel(latitude=lat), ax=axrho)
        axrho.set_xlabel("$N²$")
        dcpy.plots.set_axes_color(axrho, "C0", spine="top")
        axes.set_xlabel("Ri")
        # axes.set_title(f"lat={np.round(lat, 1)}")
        axes.text(
            x=0.95,
            y=0.93,
            s=f"{np.round(lat, 1)}N",
            transform=axes.transAxes,
            ha="right",
            va="center",
        )
        if expect_lat != 0:
            axes.set_yticklabels([])

    [dcpy.plots.linex(0.25, ax=ax) for ax in ax["lats"]]
    ax["lats"][0].set_ylabel("depth")
    ax["sst"].set_ylabel("lat")

    # ax["v"].set_ylabel("")
    # ax["v"].set_yticklabels([])

    str_time = lambda x: x.dt.round("D").dt.strftime("%Y-%m-%d").values

    adcp_str = "not found"
    if "EXPOCODE" in adcp.attrs and "CRUISE_NAME" in adcp.attrs:
        adcp_str = (
            adcp.attrs["EXPOCODE"].strip() + "; " + adcp.attrs["CRUISE_NAME"].strip()
        )
    elif "cruise_id" in adcp.attrs:
        import re

        for strings in adcp.cruise_id.split():
            match = re.search("EXPOCODE=(.*)", strings)
            if match is not None:
                adcp_str = match.string[9:]
                break

    dcpy.plots.label_subplots(
        list(ax.values())[:4],
        labels=[
            "OISST",
            "$U_z² + V_z² - 4N²$ | $ρ$ contours",
            "$U_z² - 2 N²$ | $U$ contours",
            "$V_z² - 2 N²$ | $V$ contours",
        ],
        fontsize="medium",
        y=0.05,
        x=0.015,
        backgroundcolor=[0.7, 0.7, 0.7, 0.4],
    )

    f.suptitle(
        f"CTD: {ctd.attrs['EXPOCODE'].strip()} | ADCP: {adcp_str} | {str_time(ctd.time[0])} - {str_time(ctd.time[-1])}"
    )

    f.set_size_inches((12, 7.5))

    return f, ax


def plot_row(ctd, adcp, binned, oisst, ax, expected_lat):
    """
    Intended for use with OSM20 plot or paper figure.
    Plots one row for a cruise.
    """

    # expected_lat = [0, 1, 3, 4]

    # SST
    obs_days = np.unique(ctd.sel(latitude=expected_lat[-1], method="nearest").time)
    print(obs_days)
    cax = dcpy.plots.cbar_inset_axes(ax["sst"])
    (
        oisst.sel(time=obs_days, method="nearest")
        .squeeze()
        .sel(lon=slice(-120, -105), lat=slice(-2, 6))
        .plot(
            # levels=21,
            ax=ax["sst"],
            robust=True,
            cmap=mpl.cm.RdYlBu_r,
            # add_colorbar=False,
            add_labels=False,
            cbar_kwargs={"orientation": "horizontal", "cax": cax},
        )
    )

    ax["sst"].set_ylabel("")

    stations = ctd.sel(latitude=expected_lat, method="nearest")
    ax["sst"].plot(
        stations.longitude.broadcast_like(stations.latitude),
        stations.latitude,
        marker="o",
        color="w",
        markersize=8,
    )

    ax["sst"].plot(
        ctd.longitude.broadcast_like(ctd.latitude),
        ctd.latitude,
        marker="o",
        color="k",
        markersize=4,
    )

    plotted_lats = []
    axes_rho = []
    # Ri profiles
    for axes, lat, expect_lat in zip(
        ax["lats"],
        binned.latitude.sel(latitude=expected_lat, method="nearest").values,
        expected_lat,
    ):

        # print([lat, expect_lat])
        # print(plotted_lats)
        # import IPython; IPython.core.debugger.set_trace()
        if np.abs(lat - expect_lat) > 0.3 and np.round(lat, 2) not in plotted_lats:
            print(f"skipping {expect_lat}")
            axes.remove()
            continue

        plotted_lats.append(lat)

        binned.Ri.sel(latitude=lat).plot(
            ax=axes,
            y="depth",
            xscale="log",
            yincrease=False,
            xlim=(0.1, 4),
            ylim=(100, 0),
            marker=".",
            color="k",
            _labels=False,
        )

        axrho = axes.twiny()

        rem = ctd.sizes["pressure"] % 3
        if rem != 0:
            dens = ctd.density.isel(pressure=slice(-rem))
        else:
            dens = ctd.density
        N2 = 9.81 / 1025 * dens.coarsen(pressure=3).mean().differentiate("pressure")
        # ctd.density - ctd.density.bfill("pressure").isel(pressure=0)
        N2.sel(latitude=lat, pressure=slice(0, 150)).plot(
            ax=axrho,
            y="pressure",
            ylim=(100, 0),
            _labels=False,
            xlim=(1e-6, 5e-4),
            xscale="log",
        )
        axrho.set_xticks([1e-6, 1e-5, 1e-4])
        dcpy.plots.liney(binned.mld.sel(latitude=lat), ax=axrho)
        axrho.set_xlabel("$N²$")
        dcpy.plots.set_axes_color(axrho, "C0", spine="top")
        axes.set_xlabel("$Ri$")
        # axes.set_title(f"lat={np.round(lat, 1)}")
        axes.text(
            x=0.95,
            y=0.93,
            s=f"{np.round(np.abs(lat), 1)}N",
            transform=axes.transAxes,
            ha="right",
            va="center",
        )
        if expect_lat != 0:
            axes.set_yticklabels([])

        axes_rho.append(axrho)
    [dcpy.plots.linex(0.25, ax=ax) for ax in ax["lats"]]
    ax["lats"][0].set_ylabel("depth")
    ax["sst"].set_ylabel("lat")

    # ax["v"].set_ylabel("")
    # ax["v"].set_yticklabels([])

    str_time = lambda x: x.dt.round("D").dt.strftime("%Y-%m-%d").values

    adcp_str = "not found"
    if "EXPOCODE" in adcp.attrs and "CRUISE_NAME" in adcp.attrs:
        adcp_str = (
            adcp.attrs["EXPOCODE"].strip() + "; " + adcp.attrs["CRUISE_NAME"].strip()
        )
    elif "cruise_id" in adcp.attrs:
        import re

        for strings in adcp.cruise_id.split():
            match = re.search("EXPOCODE=(.*)", strings)
            if match is not None:
                adcp_str = match.string[9:]
                break

    return axes_rho


def process_adcp_file(adcp_file: str, cruises):
    """ processes an adcp adcp_file; finds matching CTD section. """

    expocode = os.path.split(adcp_file)[-1].split("_")[0].strip()
    print(expocode)
    ctd = find_cruise(cruises, expocode)
    if ctd is None:
        return [
            None,
        ] * 4
    # ctd["density"] = dcpy.eos.pden(ctd.salinity, ctd.temperature, ctd.pressure)
    adcp = read_adcp(adcp_file, -110)
    ctd = ctd.sortby("time")
    ctd, adcp = trim_ctd_adcp(ctd, adcp)
    binned = grid_ctd_adcp(ctd, adcp)

    return expocode, ctd, adcp, binned
