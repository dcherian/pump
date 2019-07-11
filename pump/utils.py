import dask
import dask.distributed
import ncar_jobqueue


def build_cluster():
    ''' Builds and returns cluster, client objects.

    Returns
    -------
    cluster: NCARCluster object
    client: distributed.Client object
    '''

    cluster = ncar_jobqueue.NCARCluster()
    client = dask.distributed.Client(cluster)

    return cluster, client


def read_pop(files):
    def preprocess(ds):
        return ds[['VVEL', 'TEMP']].reset_coords(drop=True)

    ds = xr.open_mfdataset(files, parallel=True, preprocess=preprocess)
    file0 = xr.open_dataset(files[0])
    ds.update(file0[['TLONG', 'TLAT', 'ULONG', 'ULAT']])
    file0.close()

    return ds
