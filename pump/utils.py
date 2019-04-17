import dask
import dask.distributed
import ncar_jobqueue

def build_cluster():
    cluster = ncar_jobqueue.NCARCluster()
    client = dask.distributed.Client(cluster)

    return cluster, client
