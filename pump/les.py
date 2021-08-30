import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def write_to_txt(ds, outdir, prefix, interleave, t0=None):

    f = debug_plots(ds.drop_vars(["XG", "YG"]))
    f.suptitle(prefix[:-1], y=1.02)
    f.savefig(f"{outdir}/{prefix}_plot.png", dpi=150)

    if interleave:
        assert t0 is not None
        ds = interleave_time(ds, t0)

    ds = ds.cf.transpose("time", ...)
    for var in ds:
        ds[var].data.ravel().tofile(
            f"{outdir}/{prefix}{var}.txt", sep="\n", format="%15.7f"
        )


def interleave_time(ds, t0):
    """Interleaves time with values. For colpsh and colfrc files."""

    time = ds.cf["time"]

    assert time.ndim == 1

    ds.coords["reltime"] = (
        time.dims[0],
        (ds.cf["time"].data - t0).astype("timedelta64[s]").astype(float),
    )

    def _interleave(da):
        stacked = np.hstack([da.reltime.data, da.squeeze().data])
        return xr.DataArray(stacked, dims="zt")

    return ds.map(lambda x: x.cf.groupby("time", squeeze=False).apply(_interleave))


def interpolate(ds, newz):
    """Interpolate to les Z grid"""

    # dcpy.interpolate.UnivariateSpline(subset, "z_rho").interp(newz.values)
    interped = ds.cf.interp(
        Z=newz.values, method="slinear", kwargs={"fill_value": "extrapolate"}
    ).compute()

    # check that there are no NaNs
    for k, v in interped.isnull().sum().items():
        assert v.data == 0, "NaNs detected after interpolation"

    return interped


def debug_plots(ds):

    varnames = sorted(set(ds.data_vars) - {"time"})

    if "time" in ds.cf.dims:
        sharex = True
    else:
        sharex = False

    f, ax = plt.subplots(len(varnames), 1, sharex=sharex, constrained_layout=True)
    for var, axis in zip(varnames, ax.squeeze()):
        if var == "time":
            continue
        if (
            ds[var].ndim == 1
            and "time" not in ds[var].dims
            and "time_index" not in ds[var].dims
        ):
            ds[var].plot(y="RC", ax=axis)
        elif "time" in ds[var].dims:
            ds[var].cf.plot(x="time", ax=axis)
        else:
            ds[var].cf.plot(x="time_index", ax=axis)

    f.set_size_inches((6, 8))

    return f
