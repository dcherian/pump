import dask
import dask.distributed

try:
    import ncar_jobqueue
except (ImportError, TypeError, RuntimeError):
    ncar_jobqueue = False

import numpy as np
import xarray as xr


def build_cluster():
    """Builds and returns cluster, client objects.

    Returns
    -------
    cluster: NCARCluster object
    client: distributed.Client object
    """

    if not ncar_jobqueue:
        raise ImportError("Could not import ncar_jobqueue succesfully.")

    cluster = ncar_jobqueue.NCARCluster()
    client = dask.distributed.Client(cluster)

    return cluster, client


def read_pop(files):
    def preprocess(ds):
        return ds[["VVEL", "TEMP"]].reset_coords(drop=True)

    ds = xr.open_mfdataset(files, parallel=True, preprocess=preprocess)
    file0 = xr.open_dataset(files[0])
    ds.update(file0[["TLONG", "TLAT", "ULONG", "ULAT"]])
    file0.close

    return ds


def lowpass(obj, coord, freq, cycles_per="s", order=2, use_overlap=True, debug=False):
    """
    Lowpass butterworth filter

    """
    import dcpy
    import scipy as sp

    def _process_time(time, cycles_per="s"):
        time = time.copy()
        dt = np.nanmedian(np.diff(time.values) / np.timedelta64(1, cycles_per))

        time = np.cumsum(
            time.copy().diff(dim=time.dims[0]) / np.timedelta64(1, cycles_per)
        )

        return dt, time

    if obj[coord].dtype.kind == "M":
        dx, x = _process_time(obj[coord], cycles_per)
    else:
        dx = np.diff(obj[coord][0:2].values)

    b, a = sp.signal.butter(order, freq * dx / (1 / 2), btype="low")

    def _wrapper(data):
        return sp.signal.filtfilt(b, a, data, method="gust")

    if use_overlap:
        # 1e-2 fails xarray.testing.assert_allclose, 1e-3 and smaller are fine
        padlen = dcpy.ts.EstimateImpulseResponseLength(b, a, eps=1e-3)
        axis = obj.get_axis_num(coord)
        if debug:
            print(f"padding with length {padlen}")
        da = obj.chunk({coord: 40 * padlen}).data
        result = obj.copy(
            data=da.map_overlap(_wrapper, boundary="none", depth={axis: padlen + 1})
        )

    else:
        result = xr.apply_ufunc(
            _wrapper,
            obj,
            dask="parallelized",
            input_core_dims=[[coord]],
            output_core_dims=[[coord]],
            output_dtypes=[np.float64],
        )
    return result


def read_gcm1_zarr_subset(gcm1):
    full = xr.open_zarr(
        "../TPOS_MITgcm_1_hb/sections/gcm1-sections-retry.zarr", consolidated=True
    )
    jq = xr.open_zarr(
        "../TPOS_MITgcm_1_hb/sections/gcm1-sections-retry-Jq-2.zarr", consolidated=True
    )
    dens = xr.open_zarr(
        "../TPOS_MITgcm_1_hb/sections/gcm1-sections-dens.zarr", consolidated=True
    )
    full["time"] = gcm1.tao.time
    full["Jq"] = jq.Jq
    # assert(full.sizes["time"] == gcm1.full.sizes["time"])
    full["dens"] = dens.dens

    return full
