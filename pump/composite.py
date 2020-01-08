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


def pick(whitelist, d):
    return keyfilter(lambda k: k in whitelist, d)


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


def _get_tiv_extent_single_period(data, iy0, debug=False):

    indexes, properties = sp.signal.find_peaks(-data, prominence=0.1)
    indexes = np.array(indexes)
    # only keep locations where sst anomaly is negative
    indexes = indexes[data[indexes] < 0]
    if debug:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(data)
        dcpy.plots.linex(indexes)
        plt.title("_get_tiv_extent_single_period")
        print(indexes)

    # pick the two closest to the "center"
    final_indexes = [0, 0]

    south = indexes[indexes < iy0]
    north = indexes[indexes > iy0]
    final_indexes[0] = south[np.argmin(np.abs(south - iy0))]
    final_indexes[1] = north[np.argmin(np.abs(north - iy0))]

    return np.array([final_indexes])


def get_tiv_extent(data, dim="latitude", debug=False):

    iy0 = data.where(np.abs(data.latitude) < 4).argmax(dim)
    y0 = data.latitude[iy0]

    indexes = xr.apply_ufunc(
        _get_tiv_extent_single_period,
        data,
        iy0,
        vectorize=True,  # loops with numpy over core dim?
        dask="parallelized",  # loop with dask
        input_core_dims=[[dim], []],
        output_core_dims=[["loc"]],  # added a new dimension
        output_dtypes=[np.int32],  # required for dask
        kwargs=dict(debug=debug),
    )

    indexes["loc"] = ["bot", "top"]
    indexes = indexes.reindex(loc=["bot", "cen", "top"], fill_value=0)
    indexes.loc[{"loc": "cen"}] = iy0

    return data.latitude[indexes]


def _get_latitude_reference(data, debug=False):
    reference = get_tiv_extent(data.sel(latitude=slice(-10, 10)), debug=debug)

    y = xr.full_like(data, fill_value=np.nan)
    y.loc[:, reference.sel(loc="bot")] = -1
    y.loc[:, reference.sel(loc="cen")] = 0
    y.loc[:, reference.sel(loc="top")] = +1
    y = y.interpolate_na("latitude", fill_value="extrapolate")

    if debug:
        import dcpy

        data = data.copy().assign_coords(y=y)
        f, ax = plt.subplots(2, 1, sharey=True, constrained_layout=True)
        data.plot.line(hue="period", ax=ax[0])
        dcpy.plots.linex(reference.values.flat, ax=ax[0])
        data.plot.line(x="y", hue="period", ax=ax[1])
        dcpy.plots.linex([-1, 0, 1], ax=ax[1])

    return y


def get_warm_anom_index(ds):
    # squeeze out and drop longitude dim
    ds = ds.unstack().squeeze().reset_coords(drop=True)
    mean = ds.mean("latitude")

    if mean.count == 0:
        return np.nan

    idx = mean.argmax("time")
    return idx


def get_y_reference(theta, periods=None, debug=False):
    import IPython

    IPython.core.debugger.set_trace()

    if periods is not None:
        subset = theta.where(theta.period.isin(periods), drop=True)
    else:
        subset = theta
    # sensitive to changing this to subset.mean("time")
    mean_theta = subset.mean("time")
    anom = subset.rolling(latitude=10, center=True, min_periods=1).mean() - mean_theta
    # mean.plot.line(y="latitude", color='k')

    # use phase=180 to determine warm extent
    # t180 = (
    #    anom.where((np.abs(subset.tiw_phase - 180) < 10), drop=True)
    #    .groupby("period")
    #    .mean("time")
    # )

    # use sst warm anomaly to determine warm extent
    indexes = dict()
    idx0 = []
    for lon in anom.longitude.values:
        indexes[lon] = []
        grouped = anom.sel(longitude=lon, latitude=slice(-2, 2)).groupby("period")
        for per, group in grouped:
            if per not in periods:
                continue
            indexes[lon].append(get_warm_anom_index(group))
        idx0.append(
            xr.DataArray(
                [ind[0] for ind in grouped._group_indices],
                dims=["period"],
                coords={"period": grouped._unique_coord},
            ).sel(period=periods)
        )

    import IPython; IPython.core.debugger.set_trace()

    computed = dask.compute(list(indexes.values()))[0]

    warm_index = xr.combine_nested(computed, ["longitude", "period"]).assign_coords(
        {"period": periods, "longitude": anom.longitude}
    )
    idx0 = xr.concat(idx0, anom.longitude)
    warm_index += idx0

    t180 = anom.isel(time=warm_index.values)
    t180 = t180.copy(
        data=sp.signal.detrend(
            t180.values, type="linear", axis=t180.get_axis_num("latitude")
        )
    )

    if debug:
        plt.figure()
        t180.squeeze().plot.line(hue="time")

    yref = _get_latitude_reference(t180, debug=debug)
    ynew = xr.full_like(theta, fill_value=np.nan).compute()
    for lon in ynew.longitude.values:
        for period in np.unique(yref.period.values):
            ynew.loc[{"longitude": lon}] = xr.where(
                theta.sel(longitude=lon).period == period,
                yref.sel(longitude=lon).swap_dims({"time": "period"}).sel(period=period),
                ynew.sel(longitude=lon),
            )

    return ynew


@guvectorize(
    [(float64[:], float64[:], float64[:], float64[:])],
    "(m),(m),(n)->(n)",
    nopython=True,
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
        #exclude_dims=set(["latitude"]),  # since size of dimension is changing
        # vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float64],
    )

    result[coord] = new_coord

    return result


def make_composite(data):
    interped = to_uniform_grid(data, "yref", np.arange(-4, 4, 0.01))
    phase_grouped = interped.groupby_bins("tiw_phase", np.arange(0, 360, 5))
    data_vars = data.data_vars
    composite = {name: xr.Dataset(attrs={"name": name}) for name in data_vars}
    attr_to_name = {"mean": "avg_full", "std": "dev"}

    mean_yref = data.yref.groupby("period").mean().mean("period")
    mean_lat = xr.DataArray(
        np.interp(interped.yref, mean_yref, mean_yref.latitude),
        name="latitude",
        dims=["yref"],
    )

    for attr in ["mean", "std"]:
        computed = getattr(phase_grouped, attr)()
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

    t180 = full.time.where(
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
    if t180.notnull():
        dcpy.plots.linex(
            t180, ax=pick(["sst", "sst_yref", "vavg"], ax).values(), zorder=10
        )
