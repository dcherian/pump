import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def _get_tiv_extent_single_period(data, debug=False):
    import scipy as sp

    indexes, properties = sp.signal.find_peaks(-data, prominence=0.5)

    if debug:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(data)
        dcpy.plots.linex(indexes)
        print(indexes)

    assert len(indexes) == 2
    return indexes


def get_tiv_extent(data, dim="latitude", debug=False):

    iy0 = data.where(np.abs(data.latitude) < 2.5).argmax(dim)
    y0 = data.latitude[iy0]

    indexes = xr.apply_ufunc(
        _get_tiv_extent_single_period,
        data,
        vectorize=True,  # loops with numpy over core dim?
        dask="parallelized",  # loop with dask
        input_core_dims=[[dim]],
        output_core_dims=[["loc"]],  # added a new dimension
        output_dtypes=[np.int32],  # required for dask
        kwargs=dict(debug=debug),
    )

    indexes["loc"] = ["bot", "top"]
    indexes = indexes.reindex(loc=["bot", "cen", "top"], fill_value=0)
    indexes.loc[:, "cen"] = iy0
    return data.latitude[indexes]


def get_latitude_reference(data, debug=False):
    reference = get_tiv_extent(data.sel(latitude=slice(-6, 6)), debug=debug)

    y = xr.full_like(data, fill_value=np.nan)
    y.loc[:, reference.sel(loc="bot")] = -1
    y.loc[:, reference.sel(loc="cen")] = 0
    y.loc[:, reference.sel(loc="top")] = +1
    y = y.interpolate_na("latitude", fill_value="extrapolate")

    if debug:
        data = data.copy().assign_coords(y=y)
        f, ax = plt.subplots(2, 1, sharey=True, constrained_layout=True)
        data.plot.line(hue="period", ax=ax[0])
        dcpy.plots.linex(reference.values.flat, ax=ax[0])
        data.plot.line(x="y", hue="period", ax=ax[1])
        dcpy.plots.linex([-1, 0, 1], ax=ax[1])

    return y


def get_y_reference(theta, periods=None, debug=False):

    if periods is not None:
        subset = theta.where(theta.period.isin(periods), drop=True)
    else:
        subset = theta
    mean_theta = theta.mean("time")  # sensitive to changing this to subset.mean("time")
    anom = subset - mean_theta
    # mean.plot.line(y="latitude", color='k')
    t180 = (
        anom.where((np.abs(subset.tiw_phase - 180) < 10), drop=True)
        .groupby("period")
        .mean("time")
    )

    # plt.figure(); t180.plot.line(hue="period")

    yref = get_latitude_reference(t180, debug=debug)
    ynew = xr.full_like(theta, fill_value=np.nan)
    for period in np.unique(yref.period.values):
        ynew = xr.where(theta.period == period, yref.sel(period=period), ynew)

    return ynew
