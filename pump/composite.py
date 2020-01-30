import dask
import dcpy
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy as sp
import xarray as xr

import toolz

# from .calc import tiw_avg_filter_v

from toolz import keyfilter
from numba import guvectorize, float32, float64


good_periods = {
    # -110: [1, 2, 3, 4, 5, 7, 11, 13, 14,],  # 16 is borderline
    -110: [1, 2, 3, 4, 5],  # 16 is borderline
    # -125: [2, 3, 4, 5, 6, 11, 12, 13],  # 10 is borderline
    -125: [2, 3, 4],  # 10 is borderline
    -140: [2, 3, 4, 5, 6, 10, 11, 12, 13],
    -155: [1, 2, 3, 10],
}


def pick(whitelist, d):
    return keyfilter(lambda k: k in whitelist, d)


def detrend(data, dim):
    def _wrapper(data, type):
        out = np.full_like(data, fill_value=np.nan)
        good = ~np.isnan(data)
        if np.sum(good) > 1:
            out[good] = sp.signal.detrend(data[good], type=type, axis=-1)
        return out

    return xr.apply_ufunc(
        _wrapper,
        data,
        vectorize=True,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        kwargs={"type": "linear"},
        dask="parallelized",
        output_dtypes=[data.dtype],
    )


class Composite:

    data = dict()

    def __init__(self, comp_dict):
        self.data = comp_dict

    def __getattr__(self, key):
        return self.data[key]

    def get(self, key, filter):
        var = getattr(self, key)
        if filter:
            data = var.avg
        else:
            data = var.avg_full

        return data

    def plot(self, variable, filter=True, ax=None, **kwargs):

        if ax is None:
            ax = plt.gca()
        data = self.get(variable, filter)
        defaults = dict(y="mean_lat", robust=True, x="tiw_phase_bins")
        [kwargs.setdefault(k, v) for k, v in defaults.items()]
        cbar_kwargs = {"label": f"{data.name} variable"}
        handle = data.plot(ax=ax, **kwargs, **cbar_kwargs)
        return handle

    def overlay_contours(self, variable, filter=False, ax=None, **kwargs):

        if ax is None:
            ax = plt.gca()
        overlay = self.get(variable, filter)
        defaults = dict(
            y="mean_lat",
            robust=True,
            x="tiw_phase_bins",
            cmap=mpl.cm.RdBu_r,
            linewidths=1.5,
            levels=9,
        )
        [kwargs.setdefault(k, v) for k, v in defaults.items()]
        handle = overlay.plot.contour(ax=ax, add_colorbar=False, **kwargs)
        return handle

    def __setitem__(self, key, value):
        self.data[key] = value

    def load(self, keys=None):
        if keys is None:
            keys = self.data.keys()
        if isinstance(keys, str):
            keys = [keys]

        tasks = []
        for key in keys:
            tasks.append(self.data[key])

        results = dask.compute(*tasks)
        self.data.update(dict(zip(keys, results)))


def tiw_avg_filter_v(v):
    import xfilter

    v = xfilter.lowpass(
        v.sel(depth=slice(-10, -80)).mean("depth"),
        coord="time",
        freq=1 / 10.0,
        cycles_per="D",
        method="pad",
        gappy=False,
        num_discard=0,
    )

    if v.count() == 0:
        raise ValueError("No good data in filtered depth-averaged v.")

    v.attrs["long_name"] = "v: (10, 80m) avg, 10d lowpass"

    return v


def _get_tiv_extent_single_period(data, iy0, debug_ax, debug=False):

    prom = 0.1
    indexes, properties = sp.signal.find_peaks(-data, prominence=prom)
    indexes = np.array(indexes)

    # only keep locations where sst anomaly is negative
    # threshold = 0
    # new_indexes = indexes[data[indexes] < threshold]
    # while len(new_indexes[new_indexes < iy0]) == 0:
    #     threshold -= 0.05
    #     new_indexes = indexes[data[indexes] < threshold]
    #     if threshold > 1:
    #         raise ValueError("Infinite loop?")
    # indexes = new_indexes

    if debug:
        import matplotlib.pyplot as plt

        if debug_ax is None:
            plt.figure()
            debug_ax = plt.gca()

        debug_ax.plot(data)

    pos_indexes = indexes[indexes > iy0]
    while len(pos_indexes) == 0:
        # print("iterating north")
        prom -= 0.01
        new_indexes, _ = sp.signal.find_peaks(-data, prominence=prom)
        pos_indexes = np.array(new_indexes)[new_indexes > iy0]
        if prom < 0.01:
            raise ValueError("No northern location found")

    neg_indexes = indexes[indexes < iy0]
    while len(neg_indexes) == 0:
        # print("iterating south")
        prom -= 0.01
        new_indexes, _ = sp.signal.find_peaks(-data, prominence=prom)
        neg_indexes = np.array(new_indexes)[new_indexes < iy0]
        if prom < 0.01:
            raise ValueError("No southern location found")

    indexes = np.sort(np.concatenate([neg_indexes, pos_indexes]))

    # prevent too "thin" vortices
    # added for 125W, period=5  # not true as of Jan 27, 2020
    indexes = indexes[np.abs(indexes - iy0) > 10]

    if debug:
        dcpy.plots.linex(indexes, ax=debug_ax)
        dcpy.plots.linex(iy0, color="r", ax=debug_ax)
        print(indexes)

    # pick the two closest to the "center"
    final_indexes = [0, 0]

    south = indexes[indexes < iy0]
    north = indexes[indexes > iy0]
    final_indexes[0] = south[np.argmin(np.abs(south - iy0))]
    final_indexes[1] = north[np.argmin(np.abs(north - iy0))]

    return np.array([final_indexes])


def get_tiv_extent(data, kind, dim="latitude", debug=False, savefig=False):

    if kind == "warm":
        # data = data - 0.15
        iy0 = data.where(np.abs(data.latitude) < 3).argmax(dim)
    elif kind == "cold":
        data = data.squeeze()
        near_eq = data.sel(latitude=slice(-3, 3))
        data = np.abs((data / near_eq.min("latitude")) - 0.1)
        iy0 = data.where((data.latitude <= 3) & (data.latitude >= -3)).argmax(
            "latitude"
        )
    else:
        raise ValueError(f"'kind' must be 'warm' or 'cold'. Received {kind} instead")

    nperiod = data.sizes["period"]
    if debug:
        f, ax = plt.subplots(2, np.int(np.ceil(nperiod / 2)), sharey=True, sharex=True)
        ax = np.array(ax.flat)[:nperiod]
        f.suptitle("_get_tiv_extent_single_period")
        [aa.set_title(period) for aa, period in zip(ax, data.period.values)]
    else:
        ax = np.array([None] * nperiod)

    indexes = xr.apply_ufunc(
        _get_tiv_extent_single_period,
        data,
        iy0,
        ax,
        vectorize=True,
        dask="parallelized",
        input_core_dims=[[dim], [], []],
        output_core_dims=[["loc"]],  # added a new dimension
        output_dtypes=[np.int32],
        output_sizes={"loc": 2},
        kwargs=dict(debug=debug),
    )

    indexes["loc"] = ["bot", "top"]
    center = iy0.expand_dims(loc=["cen"])
    indexes = (
        xr.concat([indexes, center], "loc")
        .reindex(loc=["bot", "cen", "top"], fill_value=0)
        .compute()
    )

    if savefig:
        f.savefig(
            f"images/composite-debugging/tiv-extent-{np.abs(data.longitude.values[0])}.png",
            bbox_inches="tight",
        )

    return data.latitude[indexes]


def sst_for_y_reference_warm(anom):
    """
    use sst warm anomaly to determine warm extent
    """

    def get_warm_anom_index(ds):
        # squeeze out and drop longitude dim
        ds = ds.unstack().squeeze().reset_coords(drop=True)
        idx = ds.var("latitude").argmax("time")
        return idx

    # use sst warm anomaly to determine warm extent
    indexes = []
    grouped = anom.sel(latitude=slice(-5, 5)).groupby("period")
    for period, group in grouped:
        if anom.longitude == -155:
            if np.int(period) == 1:
                mask = (group.tiw_phase >= 220) & (group.tiw_phase <= 270)
            elif np.int(period) == 3:
                mask = (group.tiw_phase >= 220) & (group.tiw_phase <= 270)
            elif np.int(period) == 10:
                mask = (group.tiw_phase >= 90) & (group.tiw_phase <= 180)
            else:
                mask = (group.tiw_phase >= 180) & (group.tiw_phase <= 220)

        elif anom.longitude == -140:
            if np.int(period) == 3:
                mask = (group.tiw_phase >= 90) & (group.tiw_phase <= 180)
            elif np.int(period) == 4:
                mask = (group.tiw_phase >= 180) & (group.tiw_phase <= 215)
            elif np.int(period) == 5:
                mask = (group.tiw_phase >= 225) & (group.tiw_phase <= 270)
            elif np.int(period) == 6:
                mask = (group.tiw_phase >= 180) & (group.tiw_phase <= 210)
            else:
                mask = (group.tiw_phase >= 130) & (group.tiw_phase <= 215)

        elif anom.longitude == -125:
            # if np.int(period) == 3:
            #    mask = (group.tiw_phase >= 120) & (group.tiw_phase <= 180)
            if np.int(period) == 4:
                mask = (group.tiw_phase >= 130) & (group.tiw_phase <= 170)
            else:
                mask = (group.tiw_phase >= 130) & (group.tiw_phase <= 215)

        elif anom.longitude == -110:
            if np.int(period) == 3 or np.int(period) == 5:
                mask = (group.tiw_phase >= 220) & (group.tiw_phase <= 240)
            else:
                mask = (group.tiw_phase >= 210) & (group.tiw_phase <= 240)

        else:
            raise ValueError(f"Please add mask for longitude={anom.longitude.values}")

        indexes.append(get_warm_anom_index(group.where(mask)))

    warm_index = xr.concat(dask.compute(indexes)[0], grouped._unique_coord)
    idx0 = xr.DataArray(
        [ind[0] for ind in grouped._group_indices],
        dims=["period"],
        coords={"period": grouped._unique_coord},
    )

    warm_index += idx0

    t180 = anom.squeeze().isel(time=warm_index.values).swap_dims({"time": "period"})

    return t180.expand_dims("longitude")


def sst_for_y_reference_cold(anom):
    """
    use SST cold anomaly to determine y reference
    """

    def center(x):
        mask = (x.tiw_phase >= 0) & (x.tiw_phase <= 180)
        med = x.where(mask).median("time")
        return med

    median = anom.groupby("period").map(center)

    return median  # .expand_dims("longitude")


def tiw_period_anom(x):

    x = detrend(x, "latitude")

    return x - x.mean()
    # mask = (x.tiw_phase >=90) & (x.tiw_phase <=180)
    # mean = x.where(mask).median()
    # mean = x.mean()
    # return x #  - mean


def _get_y_reference(theta, periods=None, kind="cold", debug=False, savefig=False):

    if periods is not None:
        subset = theta.where(theta.period.isin(periods), drop=True)
    else:
        subset = theta

    subset = subset.where(
        subset.period.isin(good_periods[subset.longitude.values.item()]), drop=True
    )

    # import IPython; IPython.core.debugger.set_trace()

    # ATTEMPT 1:
    # use phase=180 to determine warm extent
    # doesn't work so well because the phase calculation may not line up well
    # t180 = (
    #    anom.where((np.abs(subset.tiw_phase - 180) < 10), drop=True)
    #    .groupby("period")
    #    .mean("time")
    # )

    if kind == "warm":
        # ATTEMPT 2:
        # try to find the warm anomaly and reference to that
        # works well at -110, but not so well at -125, -140
        # sensitive to changing this to subset.mean("time")

        # TODO: need to change form time coordinate to period coordinate as for kind == "cold"
        anom = subset.groupby("period") - subset.groupby("period").mean("time")
        anom = detrend(anom, dim="latitude")
        anom = anom.rolling(latitude=21, center=True, min_periods=1).mean()
        sst_ref = sst_for_y_reference_warm(anom).copy()
        # sst_ref = sst_ref.copy(
        #    data=sp.signal.detrend(
        #        sst_ref.values, type="linear", axis=sst_ref.get_axis_num("latitude"),
        #    )
        # )

    elif kind == "cold":
        # ATTEMPT 3:
        # reference to cold anomaly and use medians instead of means to avoid "warm bias"

        anom = theta.groupby("period").apply(tiw_period_anom)
        sst_ref = sst_for_y_reference_cold(anom)

    if debug:
        nperiod = len(np.unique(anom.period))
        fg = (
            anom.squeeze()
            .groupby("period")
            .plot(
                col="period",
                col_wrap=np.int(np.ceil(nperiod / 2)),
                x="time",
                sharey=True,
                robust=True,
                cmap=mpl.cm.RdBu_r,
                add_colorbar=False,
            )
        )
        for loc, ax in zip(fg.name_dicts.flat, fg.axes.flat):
            if loc is not None:
                phase = anom.tiw_phase.where(
                    anom.period == loc["period"], drop=True
                ).squeeze()

                phase_mask = phase.round().isin([0, 90, 180, 270])
                phase_mask[-1] = True
                (
                    phase.where(phase_mask, drop=True).plot(
                        x="time",
                        ax=ax.twinx(),
                        _labels=False,
                        color="k",
                        marker="o",
                        lw=2,
                    )
                )
                # theta.where(theta.period == loc["period"], drop=True).squeeze().plot(
                #    x="time", ax=ax.twinx(), _labels=False
                # )
                (
                    sst_ref.sel(loc)
                    .squeeze()
                    .plot(y="latitude", ax=ax.twiny(), _labels=False, color="k")
                )

                if "time" in sst_ref.coords:
                    dcpy.plots.linex(
                        sst_ref.sel(loc).time.values, ax=ax, color="k", zorder=20, lw=2,
                    )
        fg.fig.suptitle(f"longitude={anom.longitude.values}", y=1.08)

    reference = get_tiv_extent(
        sst_ref.sel(latitude=slice(-8, 8)), kind=kind, debug=debug, savefig=savefig,
    )

    yref = xr.full_like(sst_ref, fill_value=np.nan)
    yref.loc[:, :, reference.sel(loc="bot")] = -1
    yref.loc[:, :, reference.sel(loc="cen")] = 0
    yref.loc[:, :, reference.sel(loc="top")] = +1
    yref = yref.interpolate_na("latitude", fill_value="extrapolate")

    if debug:
        data = sst_ref.copy().assign_coords(yref=yref).squeeze()
        f, ax = plt.subplots(2, 1, sharey=True, constrained_layout=True)
        data.plot.line(hue="period", ax=ax[0], add_legend=False)
        dcpy.plots.linex(reference.values.flat, ax=ax[0])
        data.plot.line(x="yref", hue="period", ax=ax[1], add_legend=False)
        dcpy.plots.linex([-1, 0, 1], ax=ax[1])

    ynew = xr.full_like(theta, fill_value=np.nan).compute()
    for period in np.unique(yref.period.values):
        ynew = xr.where(theta.period == period, yref.sel(period=period), ynew)
    ynew.name = "yref"
    reference.name = "reference"

    if debug:
        for loc, ax in zip(fg.name_dicts.flat, fg.axes.flat):
            if loc is not None:
                dcpy.plots.liney(
                    reference.sel(loc).squeeze().values, ax=ax, color="k", zorder=20
                )

    if savefig:
        fg.fig.savefig(
            f"images/composite-debugging/yref-sst-{np.abs(anom.longitude.values)}.png",
            bbox_inches="tight",
        )
        f.savefig(
            f"images/composite-debugging/yref-{np.abs(anom.longitude.values)}.png",
            bbox_inches="tight",
        )

    return ynew, reference


def get_y_reference(theta, periods, kind="cold", debug=False, savefig=False):
    y = []
    r = []
    for lon in theta.longitude.values:
        yy, rr = _get_y_reference(
            theta.sel(longitude=lon), periods, kind, debug, savefig
        )
        y.append(yy)
        r.append(rr)

    return xr.concat(y, "longitude"), xr.concat(r, "longitude")


@guvectorize(
    [(float64[:], float64[:], float64[:], float64[:])],
    "(m),(m),(n)->(n)",
    nopython=True,
    # nogil=True,
    cache=True,
)
def _wrap_interp(x, y, newx, out):
    out[:] = np.interp(newx, x, y)
    out[newx > np.max(x)] = np.nan
    out[newx < np.min(x)] = np.nan


def to_uniform_grid(data, coord, new_coord=np.arange(-4, 4, 0.01)):

    if isinstance(new_coord, np.ndarray):
        new_coord = xr.DataArray(new_coord, dims=[coord], name=coord)

    result = xr.apply_ufunc(
        _wrap_interp,
        data[coord],
        data,
        new_coord,
        input_core_dims=[["latitude"], ["latitude"], ["yref"]],
        output_core_dims=[["yref"]],  # order is important
        # exclude_dims=set(["latitude"]),  # since size of dimension is changing
        # vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float64],
    )

    result[coord] = new_coord

    return result


def vectorized_groupby(data, dim, group, func, kind="groupby", **kwargs):
    result = []
    for lon in data[dim].values:
        grouped = getattr(data.sel({dim: lon}), kind)(group, **kwargs)
        result.append(grouped.map(func))
    return xr.concat(result, dim)


def make_composite_multiple_masks(data, masks: dict):
    interped = to_uniform_grid(data, "yref", np.arange(-4, 4, 0.01))
    mean_yref = vectorized_groupby(
        data.yref, dim="longitude", group="period", func=lambda x: x.mean("time")
    ).mean("period")

    # TODO: remove this vectorize bit
    mean_lat = xr.apply_ufunc(
        np.interp,
        interped.yref,
        mean_yref,
        mean_yref.latitude,
        vectorize=True,
        input_core_dims=[["yref"], ["latitude"], ["latitude"]],
        output_core_dims=[["yref"]],
    )
    mean_lat.name = "latitude"

    # avoid where's annoying broadcasting behaviour
    unmasked_vars = ["sst", "dcl", "mld"]

    comp = {}
    for name in masks:
        masked = data.where(masks[name])
        for varname in unmasked_vars:
            if varname in masked:
                masked[varname] = data[varname]
        comp[name] = make_composite(
            masked.mean("depth"),
            interped=interped,
            mean_yref=mean_yref,
            mean_lat=mean_lat,
        )

    return comp


def make_composite(data, interped=None, mean_yref=None, mean_lat=None):
    # the next two paragraphs should move one level up so it only happens once instead of once per mask
    if interped is None:
        interped = to_uniform_grid(data, "yref", np.arange(-4, 4, 0.01))

    # mean_yref = data.yref.groupby("period").mean().mean("period")
    if mean_yref is None:
        mean_yref = vectorized_groupby(
            data.yref, dim="longitude", group="period", func=lambda x: x.mean("time")
        ).mean("period")

    if mean_lat is None:
        # TODO: remove this vectorize bit
        mean_lat = xr.apply_ufunc(
            np.interp,
            interped.yref,
            mean_yref,
            mean_yref.latitude,
            vectorize=True,
            input_core_dims=[["yref"], ["latitude"], ["latitude"]],
            output_core_dims=[["yref"]],
        )
        mean_lat.name = "latitude"

    # phase_grouped = interped.groupby_bins("tiw_phase", np.arange(0, 360, 5))
    data_vars = set(data.data_vars) | set(["mld", "dcl"])
    composite = {name: xr.Dataset(attrs={"name": name}) for name in data_vars}

    # mean_lat = xr.DataArray(
    #    np.interp(interped.yref, mean_yref, mean_yref.latitude),
    #    name="latitude",
    #    dims=["yref"],
    # )

    attr_to_name = {"mean": "avg_full", "std": "dev"}
    for attr in ["mean", "std"]:
        computed = vectorized_groupby(
            interped,
            dim="longitude",
            group="tiw_phase",
            func=lambda x: getattr(x, attr)("time"),
            kind="groupby_bins",
            bins=np.arange(0, 360, 5),
        )
        # computed = getattr(phase_grouped, attr)()
        for name in data_vars:
            composite[name][attr_to_name[attr]] = computed[name]
            composite[name]["period"] = np.unique(interped.period)
            # composite[name] = composite[name].transpose("yref", "tiw_phase_bins", "period")

    for name in data_vars:
        composite[name] = composite[name].assign_coords(mean_lat=mean_lat)
        composite[name]["err"] = composite[name].dev / np.sqrt(
            len(composite[name].period)
        )
        is_significant = np.abs(composite[name]).avg_full >= 1.96 * composite[name].err
        composite[name]["avg"] = composite[name].avg_full.where(is_significant)

    return Composite(composite)


def test_composite_algorithm(full, tao, period, debug=False):

    assert "sst" in full

    sst = full.sst - full.sst.mean("time")
    ynew = get_y_reference(full.sst, periods=period, debug=debug)
    sst = sst.assign_coords(yref=ynew)

    vavg = tiw_avg_filter_v(tao.v).where(full.period == period, drop=True)

    sst_ref = full.time.where(
        np.abs(full.tiw_phase.where(full.period == period, drop=True) - 180) < 10
    ).mean("time")

    f = plt.figure(constrained_layout=True)
    gs = f.add_gridspec(2, 2)
    ax = {}
    ax["sst"] = f.add_subplot(gs[0, 0])
    ax["sst_yref"] = f.add_subplot(gs[0, 1], sharex=ax["sst"])
    ax["vavg"] = f.add_subplot(gs[1, 0], sharex=ax["sst"])

    (
        sst.where(sst.period == period, drop=True).plot(
            x="time", ax=ax["sst"], robust=True
        )
    )
    (
        sst.where(ynew.notnull(), drop=True).plot(
            y="yref", x="time", ax=ax["sst_yref"], robust=True
        )
    )

    vavg.plot(x="time", ax=ax["vavg"])
    ax["vavg"].axhline(0)
    [axx.set_title("") for axx in ax.values()]
    if sst_ref.notnull():
        dcpy.plots.linex(
            sst_ref, ax=pick(["sst", "sst_yref", "vavg"], ax).values(), zorder=10
        )
