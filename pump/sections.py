import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
import xarray as xr

import dcpy


def read_adcp(filename, longitude, debug=False):
    adcp = xr.load_dataset(filename)

    if "lat" in adcp:
        adcp = adcp.rename({"lat": "latitude", "lon": "longitude"})

    if "depth_cell" in adcp.dims:
        xr.testing.assert_allclose(
            adcp.depth.diff("time"), xr.zeros_like(adcp.depth.diff("time"))
        )
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

    return binned


def get_bins_around_levels(levels):
    dz = np.diff(levels)
    return np.append(
        np.insert((levels[:-1] + levels[1:]) / 2, 0, levels[0] - dz[0] / 2),
        levels[-1] + dz[-1] / 2,
    )


def plot_section(ctd, adcp, binned, oisst, ladcp=None):

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

    gs2 = gsparent[1].subgridspec(1, 8)
    ax["lats"] = [f.add_subplot(gs2[nn]) for nn in range(8)]

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
            vmax=27,
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
        y="depth", yincrease=False, cmap=mpl.cm.RdBu_r, add_colorbar=True, norm=norm,
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

    expected_lat = [0, 0.5, 1, 1.5, 2, 3, 4, 5]
    plotted_lats = []
    # Ri profiles
    for axes, lat, expect_lat in zip(
        ax["lats"],
        binned.latitude.sel(latitude=expected_lat, method="nearest").values,
        expected_lat,
    ):
        # import IPython; IPython.core.debugger.set_trace()
        if (
            np.abs(lat - expect_lat) > 0.3
            and np.round(lat, 1) != 3.5
            and np.round(lat, 1) not in plotted_lats
        ):
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
            _labels=False,
        )
        axes.set_xlabel("Ri")
        axes.set_title(f"lat={np.round(lat, 1)}")
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
