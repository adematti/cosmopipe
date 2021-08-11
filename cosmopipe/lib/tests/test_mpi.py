import numpy as np

from cosmopipe.lib import setup_logging, mpi
from cosmopipe.lib.utils import TaskManager, MemoryMonitor


def test_mpi_sum():

    comm = mpi.CurrentMPIComm.get()
    for npfun,mpifun in [(np.sum,mpi.sum_array),(np.mean,mpi.mean_array),(np.prod,mpi.prod_array),(np.min,mpi.min_array),(np.max,mpi.max_array)]:
        array = None
        if comm.rank == 0:
            array = np.random.uniform(size=(5000,18,2))
        broadcast = mpi.broadcast_array(array)
        scattered = mpi.scatter_array(array)
        for axis in [(0,),(0,1)]:
            with MemoryMonitor(msg='serial  ') as mem:
                for i in range(100):
                    ref = npfun(broadcast,axis=axis)
            with MemoryMonitor(msg='parallel') as mem:
                for i in range(100):
                    val = mpifun(scattered,axis=axis)
            assert np.allclose(val,ref), '{} {}, {} {}'.format(npfun,mpifun,ref,val)


def test_mpi_argmin():

    comm = mpi.CurrentMPIComm.get()
    for npfun,mpifun in [(np.argmin,mpi.argmin_array),(np.argmax,mpi.argmax_array)]:

        array = None
        if comm.rank == 0:
            array = np.random.uniform(size=(5000,18,2))
        broadcast = mpi.broadcast_array(array)
        scattered = mpi.scatter_array(array)
        axis = 0
        with MemoryMonitor(msg='serial  ') as mem:
            for i in range(100):
                ref = npfun(broadcast,axis=axis)
        minref = np.take_along_axis(broadcast,np.expand_dims(ref,axis=axis),axis=axis)[0]
        with MemoryMonitor(msg='parallel') as mem:
            for i in range(100):
                argmin,rank = mpifun(scattered,axis=axis)
        mask = rank == comm.rank
        min = np.take_along_axis(scattered,np.expand_dims(argmin*mask,axis=axis),axis=axis)
        min = mpi.gather_array(min)
        if comm.rank == 0:
            min = np.take_along_axis(min,np.expand_dims(rank,axis=axis),axis=axis)
            assert np.all(min == minref), '{} {}'.format(minref, min)

        axis = None
        with MemoryMonitor(msg='serial  ') as mem:
            for i in range(100):
                ref = npfun(broadcast,axis=axis)
        minref = broadcast[np.unravel_index(ref,broadcast.shape)]
        with MemoryMonitor(msg='parallel') as mem:
            for i in range(100):
                argmin,rank = mpifun(scattered,axis=axis)
        min = scattered[np.unravel_index(argmin,scattered.shape)] if comm.rank == rank else None
        min = comm.bcast(min,root=rank)
        assert min == minref, '{} {}'.format(minref, min)


def test_mpi_sort():

    comm = mpi.CurrentMPIComm.get()
    array = None
    if comm.rank == 0:
        array = np.random.uniform(size=(3200,10))
    broadcast = mpi.broadcast_array(array)
    scattered = mpi.scatter_array(array)

    axis = 0
    with MemoryMonitor(msg='serial  ') as mem:
        for i in range(10):
            ref = np.sort(broadcast,axis=axis)

    with MemoryMonitor(msg='parallel') as mem:
        for i in range(10):
            toret = mpi.sort_array(scattered,axis=axis)

    toret = mpi.gather_array(toret)
    if comm.rank == 0:
        assert np.all(toret == ref), '{} {}'.format(ref, toret)
    #mpi.transposition_sort(array)

    for axis in [None,0,(0,1)]:
        for q in [0.5,(0.2,0.98)]:
            ref = np.quantile(broadcast,q=q,axis=axis)
            toret = mpi.quantile_array(scattered,q=q,axis=axis)
            if comm.rank == 0:
                assert np.all(toret == ref), '{} {} {} {}'.format(axis, q, ref, toret)


def test_mpi_algebra():

    comm = mpi.CurrentMPIComm.get()
    rng = np.random.RandomState()

    for a,b in [(rng.uniform(size=30),rng.uniform(size=30)),(rng.uniform(size=(20,2)),rng.uniform(size=(20,4)))]:
        ba = mpi.broadcast_array(a)
        sa = mpi.scatter_array(a)
        bb = mpi.broadcast_array(b)
        sb = mpi.scatter_array(b)
        ref = np.dot(ba.T,bb)
        toret = mpi.dot_array(sa.T,sb)
        if comm.rank == 0:
            assert np.allclose(toret,ref), '{} {}'.format(ref, toret)

        ref = np.cov(ba.T,bb.T)
        toret = mpi.cov_array(sa.T,sb.T)
        if comm.rank == 0:
            assert np.allclose(toret,ref), '{} {}'.format(ref, toret)

        ref = np.corrcoef(ba.T,bb.T)
        toret = mpi.corrcoef_array(ba.T,bb.T)
        if comm.rank == 0:
            assert np.allclose(toret,ref), '{} {}'.format(ref, toret)


    for a,b in [(rng.uniform(size=30),rng.uniform(size=30)),(rng.uniform(size=30),None)]:
        ba = mpi.broadcast_array(a)
        sa = mpi.scatter_array(a)
        bb = mpi.broadcast_array(b) if b is not None else b
        sb = mpi.scatter_array(b) if b is not None else b

        ref = np.average(ba,weights=bb)
        toret = mpi.average_array(sa,weights=sb)
        if comm.rank == 0:
            assert np.allclose(toret,ref), '{} {}'.format(ref, toret)

        ref = np.var(ba)
        toret = mpi.var_array(sa,ddof=0)
        if comm.rank == 0:
            assert np.allclose(toret,ref), '{} {}'.format(ref, toret)


if __name__ == '__main__':

    setup_logging(level='debug')
    test_mpi_sum()
    test_mpi_argmin()
    test_mpi_sort()
    test_mpi_algebra()
