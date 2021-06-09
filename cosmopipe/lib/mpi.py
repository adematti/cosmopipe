import atexit
import random

import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from mpi4py import MPI
from mpi4py.MPI import COMM_SELF

from pypescript.mpi import *
from . import utils

@CurrentMPIComm.enable
def set_common_seed(seed=None, mpicomm=None):
    if seed is None:
        if mpicomm.rank == 0:
            seed = np.random.randint(0, high=0xffffffff, size=1)
    seed = mpicomm.bcast(seed,root=0)
    np.random.seed(seed)
    random.seed(int(seed))
    return seed

@CurrentMPIComm.enable
def bcast_seed(seed=None, mpicomm=None, size=10000):
    if mpicomm.rank == 0:
        seeds = np.random.RandomState(seed=seed).randint(0, high=0xffffffff, size=size)
    return broadcast_array(seeds if mpicomm.rank == 0 else None,root=0,mpicomm=mpicomm)

@CurrentMPIComm.enable
def set_independent_seed(seed=None, mpicomm=None, size=10000):
    seed = bcast_seed(seed=seed,mpicomm=mpicomm,size=size)[mpicomm.rank]
    np.random.seed(seed)
    random.seed(int(seed))
    return seed


class MPIPool(object):
    """A processing pool that distributes tasks using MPI.
    With this pool class, the master process distributes tasks to worker
    processes using an MPI communicator.
    This implementation is inspired by @juliohm in `this module
    <https://github.com/juliohm/HUM/blob/master/pyhum/utils.py#L24>`_
    and was adapted from schwimmbad.
    Parameters
    ----------
    mpicomm : :class:`mpi4py.MPI.Comm`, optional
        An MPI communicator to distribute tasks with. If ``None``, this uses
        ``MPI.COMM_WORLD`` by default.
    """

    @CurrentMPIComm.enable
    def __init__(self, mpicomm=None, check_tasks=False):

        self.mpicomm = MPI.COMM_WORLD if mpicomm is None else mpicomm

        self.master = 0
        self.rank = self.mpicomm.Get_rank()

        #atexit.register(lambda: MPIPool.close(self))

        #if not self.is_master():
        #    # workers branch here and wait for work
        #    self.wait()
        #    sys.exit(0)

        self.workers = set(range(self.mpicomm.size))
        self.workers.discard(self.master)
        self.size = self.mpicomm.Get_size() - 1
        self.check_tasks = check_tasks

        if self.size == 0:
            raise ValueError("Tried to create an MPI pool, but there "
                             "was only one MPI process available. "
                             "Need at least two.")


    def wait(self):
        """Tell the workers to wait and listen for the master process. This is
        called automatically when using :meth:`MPIPool.map` and doesn't need to
        be called by the user.
        """
        if self.is_master():
            return

        status = MPI.Status()
        while True:
            task = self.mpicomm.recv(source=self.master, tag=MPI.ANY_TAG, status=status)

            if task is None:
                # Worker told to quit work
                break

            result = self.function(task)
            # Worker is sending answer with tag
            self.mpicomm.ssend(result, self.master, status.tag)


    def map(self, function, tasks):
        """Evaluate a function or callable on each task in parallel using MPI.
        The callable, ``worker``, is called on each element of the ``tasks``
        iterable. The results are returned in the expected order.
        Parameters
        ----------
        worker : callable
            A function or callable object that is executed on each element of
            the specified ``tasks`` iterable. This object must be picklable
            (i.e. it can't be a function scoped within a function or a
            ``lambda`` function). This should accept a single positional
            argument and return a single object.
        tasks : iterable
            A list or iterable of tasks. Each task can be itself an iterable
            (e.g., tuple) of values or data to pass in to the worker function.
        Returns
        -------
        results : list
            A list of results from the output of each ``worker()`` call.
        """

        # If not the master just wait for instructions.
        self.function = function
        #if not self.is_master():
        #    self.wait()
        #    return
        results = None
        tasks = list(tasks)

        # check
        if self.check_tasks:
            alltasks = self.mpicomm.allgather(tasks)
            tasks = np.array(alltasks[self.master])
            for t in alltasks:
                if t is not None and not np.all(np.array(t) == tasks):
                    raise ValueError('Something fishy: not the same input tasks on all ranks')

        if self.is_master():

            workerset = self.workers.copy()
            tasklist = [(tid, arg) for tid, arg in enumerate(tasks)]
            pending = len(tasklist)
            results = [None]*len(tasklist)

            while pending:
                if workerset and tasklist:
                    worker = workerset.pop()
                    taskid, task = tasklist.pop()
                    # "Sent task %s to worker %s with tag %s"
                    self.mpicomm.send(task, dest=worker, tag=taskid)

                if tasklist:
                    flag = self.mpicomm.Iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
                    if not flag:
                        continue
                else:
                    self.mpicomm.Probe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)

                status = MPI.Status()
                result = self.mpicomm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG,
                                        status=status)
                worker = status.source
                taskid = status.tag

                # "Master received from worker %s with tag %s"

                workerset.add(worker)
                results[taskid] = result
                pending -= 1
            self.close()
        else:
            self.wait()

        self.mpicomm.Barrier()
        return self.mpicomm.bcast(results,root=self.master)


    def close(self):
        """ Tell all the workers to quit."""
        if self.is_worker():
            return

        for worker in self.workers:
            self.mpicomm.send(None, worker, 0)


    def is_master(self):
        return self.rank == self.master


    def is_worker(self):
        return self.rank != self.master


    def __enter__(self):
        return self


    def __exit__(self, *args):
        self.close()


"""
class MPIPool(MPITaskManager):

    logger = logging.getLogger('MPIPool')

    @CurrentMPIComm.enable
    def __init__(self, mpicomm=None):
        super(MPIPool,self).__init__(nprocs_per_task=1,use_all_nprocs=True,mpicomm=mpicomm)
        self.__enter__()

    def wait(self):
        return self._get_tasks()

    def map(self, function, tasks):
        return super(MPIPool,self).map(function,list(tasks))

    def is_master(self):
        return self.is_root()

    def close(self):
        self.__exit__(None,None,None)
"""

@CurrentMPIComm.enable
def send_array(data, dest, tag=0, mpicomm=None):
    if not data.flags['C_CONTIGUOUS']:
        data = np.ascontiguousarray(data)
    shape_and_dtype = (data.shape, data.dtype)
    mpicomm.send(shape_and_dtype,dest=dest,tag=tag)
    mpicomm.Send(data,dest=dest,tag=tag)


@CurrentMPIComm.enable
def recv_array(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, mpicomm=None):
    shape, dtype = mpicomm.recv(source=source,tag=tag)
    data = np.empty(shape, dtype=dtype)
    mpicomm.Recv(data,source=source,tag=tag)
    return data


def _reduce_array(data, npop, mpiop, *args, mpicomm=None, axis=None, **kwargs):
    toret = npop(data,*args,axis=axis,**kwargs)
    if axis is None: axis = tuple(range(data.ndim))
    else: axis = normalize_axis_tuple(axis,data.ndim)
    if 0 in axis:
        if np.isscalar(toret):
            return mpicomm.allreduce(toret,op=mpiop)
        total = np.empty_like(toret)
        mpicomm.Allreduce(toret,total,op=mpiop)
        return total
    return toret


@CurrentMPIComm.enable
def size_array(data, mpicomm=None):
    return mpicomm.allreduce(data.size,op=MPI.SUM)


@CurrentMPIComm.enable
def shape_array(data, mpicomm=None):
    shapes = mpicomm.allgather(data.shape)
    shape0 = sum(s[0] for s in shapes)
    return (shape0,) + shapes[0][1:]


@CurrentMPIComm.enable
def sum_array(data, *args, mpicomm=None, axis=None, **kwargs):
    return _reduce_array(data,np.sum,MPI.SUM,*args,mpicomm=mpicomm,axis=axis,**kwargs)


@CurrentMPIComm.enable
def mean_array(data, *args, mpicomm=None, axis=-1, **kwargs):
    if axis is None: axis = tuple(range(data.ndim))
    else: axis = normalize_axis_tuple(axis,data.ndim)
    if 0 not in axis:
        toret = np.mean(data,*args,axis=axis,**kwargs)
    else:
        toret = sum_array(data,*args,mpicomm=mpicomm,axis=axis,**kwargs)
        N = size_array(data,mpicomm=mpicomm)/(1. if np.isscalar(toret) else toret.size)
        toret /= N
    return toret


@CurrentMPIComm.enable
def prod_array(data, *args, mpicomm=None, axis=None, **kwargs):
    return _reduce_array(data,np.prod,MPI.PROD,*args,mpicomm=mpicomm,axis=axis,**kwargs)


@CurrentMPIComm.enable
def min_array(data, *args, mpicomm=None, axis=None, **kwargs):
    return _reduce_array(data,np.min,MPI.MIN,*args,mpicomm=mpicomm,axis=axis,**kwargs)


@CurrentMPIComm.enable
def max_array(data, *args, mpicomm=None, axis=None, **kwargs):
    return _reduce_array(data,np.max,MPI.MAX,*args,mpicomm=mpicomm,axis=axis,**kwargs)


def _reduce_arg_array(data, npop, mpiargop, mpiop, *args, mpicomm=None, axis=None, **kwargs):
    arg = npop(data,*args,axis=axis,**kwargs)
    if axis is None:
        val = data[np.unravel_index(arg,data.shape)]
    else:
        val = np.take_along_axis(data,np.expand_dims(arg,axis=axis),axis=axis)[0]
    # could not find out how to do mpicomm.Allreduce([tmp,MPI.INT_INT],[total,MPI.INT_INT],op=MPI.MINLOC) for e.g. (double,int)...
    if axis is None: axis = tuple(range(data.ndim))
    else: axis = normalize_axis_tuple(axis,data.ndim)
    if 0 in axis:
        if np.isscalar(arg):
            rank = mpicomm.allreduce((val,mpicomm.rank),op=mpiargop)[1]
            argmin = mpicomm.bcast(arg,root=rank)
            return arg,rank
        #raise NotImplementedError('MPI argmin/argmax with non-scalar output is not implemented.')
        total = np.empty_like(val)
        # first decide from which rank we get the solution
        mpicomm.Allreduce(val,total,op=mpiop)
        mask = val == total
        rank = np.ones_like(arg) + mpicomm.size
        rank[mask] = mpicomm.rank
        totalrank = np.empty_like(rank)
        mpicomm.Allreduce(rank,totalrank,op=MPI.MIN)
        # f.. then fill in argmin
        mask = totalrank == mpicomm.rank
        tmparg = np.zeros_like(arg)
        tmparg[mask] = arg[mask]
        #print(mpicomm.rank,arg,mask)
        totalarg = np.empty_like(tmparg)
        mpicomm.Allreduce(tmparg,totalarg,op=MPI.SUM)
        return totalarg,totalrank

    return arg,None


@CurrentMPIComm.enable
def argmin_array(data, *args, mpicomm=None, axis=None, **kwargs):
    return _reduce_arg_array(data,np.argmin,MPI.MINLOC,MPI.MIN,*args,mpicomm=mpicomm,axis=axis,**kwargs)


@CurrentMPIComm.enable
def argmax_array(data, *args, mpicomm=None, axis=None, **kwargs):
    return _reduce_arg_array(data,np.argmax,MPI.MAXLOC,MPI.MAX,*args,mpicomm=mpicomm,axis=axis,**kwargs)


@CurrentMPIComm.enable
def partition_array(data, *args, mpicomm=None, axis=None, **kwargs):
    pass


def _bitonic_sort(data, axis=-1, kind=None, mpicomm=None, merge=None):
    # bitonic algorithm, inspired from https://stackoverflow.com/questions/13673507/bitonic-sort-mpi4py
    # number of ranks = power of 2, same data size on each rank
    d = mpicomm.size.bit_length() - 1

    def get_partner(rank, j):
        # Partner process is process with j_th bit of rank flipped
        j_mask = 1 << j
        return rank ^ j_mask

    for i in range(1, d+1):
        window_id = mpicomm.rank >> i
        for j in reversed(range(0, i)):
            bitj = (mpicomm.rank >> j) & 1
            mpicomm.Barrier()
            partner = get_partner(mpicomm.rank, j)
            new_data = np.empty_like(data)
            mpicomm.Send(data, dest = partner, tag=55)
            mpicomm.Recv(new_data, source = partner, tag=55)
            if (window_id%2 == bitj):
                #mpicomm.Recv(new_data, source = partner, tag=55)
                #mpicomm.Send(data, dest = partner, tag=55)
                data = np.split(merge(data, new_data), 2)[0]
            else:
                #mpicomm.Send(data, dest = partner, tag=55)
                #mpicomm.Recv(new_data, source = partner, tag=55)
                data = np.split(merge(data, new_data), 2)[1]
            mpicomm.Barrier()
    return data


def _transposition_sort(data, axis=-1, kind=None, mpicomm=None, merge=None):
    # number of ranks = multiple of 2, same data size on each rank
    size = mpicomm.size & ~1
    for step in range(0, size):
        if (step % 2 == 0):
            if (mpicomm.rank % 2 == 0):
                des = mpicomm.rank + 1
            else:
                des = mpicomm.rank - 1
        else:
            if (mpicomm.rank % 2 == 0):
                des = mpicomm.rank - 1
            else:
                des = mpicomm.rank + 1
        if (des >= 0 and des < size):
            new_data = np.empty_like(data)
            mpicomm.Send(data, dest = des, tag=55)
            mpicomm.Recv(new_data, source = des, tag=55)
            if mpicomm.rank < des:
                data = np.split(merge(data, new_data), 2)[0]
            else:
                data = np.split(merge(data, new_data), 2)[1]
    return data


@CurrentMPIComm.enable
def sort_array(data, axis=-1, kind=None, mpicomm=None):
    toret = np.sort(data,axis=axis,kind=kind)
    if mpicomm.size == 1:
        return toret
    if axis is None:
        data = data.flat
        axis = 0
    else:
        axis = normalize_axis_tuple(axis,data.ndim)[0]
    if axis != 0:
        return toret

    gathered = gather_array(toret,root=0,mpicomm=mpicomm)
    toret = None
    if mpicomm.rank == 0:
        toret = np.sort(gathered,axis=axis,kind=kind)
    return scatter_array(toret,root=0,mpicomm=mpicomm)

    """
    # Algorithms below are slower than the serial implementation on my laptop + they would need array padding, etc. Drop it for now.
    def merge(a, b):
        if a.ndim == 1:
            ii = np.searchsorted(a,b)
            return np.insert(a,ii,b,axis=0)
        # because search sorted does not take an "axis" argument...
        toret = np.concatenate([a,b],axis=0)
        toret.sort(axis=0,kind=kind)
        return toret

    sizepow2 = 2**(mpicomm.size.bit_length() - 1)
    sizex2 = mpicomm.size & ~1

    niterbitonic = mpicomm.size.bit_length()*(mpicomm.size.bit_length()-1)//2
    nitertrans = sizex2

    #if niterbitonic > nitertrans:
    if mpicomm.size == sizepow2:
        #toret = redistribute_array(toret, size=sizepow2)
        toret = _bitonic_sort(toret, axis=axis, kind=kind, mpicomm=mpicomm, merge=merge)

    #toret = redistribute_array(toret, size=sizex2)
    toret = _transposition_sort(toret, axis=axis, kind=kind, mpicomm=mpicomm, merge=merge)

    return toret
    """

@CurrentMPIComm.enable
def quantile_array(data, q, axis=None, overwrite_input=False, interpolation='linear', keepdims=False, mpicomm=None):
    if axis is None or 0 in normalize_axis_tuple(axis,data.ndim):
        gathered = gather_array(data,root=0,mpicomm=mpicomm)
        toret = None
        if mpicomm.rank == 0:
            toret = np.quantile(gathered,q,axis=axis,overwrite_input=overwrite_input,keepdims=keepdims)
        return broadcast_array(toret,root=0,mpicomm=mpicomm)
    return np.quantile(data,q,axis=axis,overwrite_input=overwrite_input,keepdims=keepdims)


@CurrentMPIComm.enable
def weighted_quantile_array(data, q, weights=None, mpicomm=None):
    # TODO extend to other axes
    axis = 0
    if axis is None or 0 in normalize_axis_tuple(axis,data.ndim):
        gathered = gather_array(data,root=0,mpicomm=mpicomm)
        weights = gather_array(weights,root=0,mpicomm=mpicomm)
        toret = None
        if mpicomm.rank == 0:
            toret = utils.weighted_quantile(gathered,q,weights=weights)
        return broadcast_array(toret,root=0,mpicomm=mpicomm)
    return utils.weighted_quantile(data,q,weights=weights)


@CurrentMPIComm.enable
def dot_array(a, b, mpicomm=None):
    # scatter axis is b first axis
    if b.ndim == 1:
        return sum_array(a*b,mpicomm=mpicomm)
    if a.ndim == b.ndim == 2:
        return sum_array(np.dot(a,b)[None,...],axis=0,mpicomm=mpicomm)
    raise NotImplementedError


@CurrentMPIComm.enable
def average_array(a, axis=None, weights=None, returned=False, mpicomm=None):
    # TODO: allow several axes
    if axis is None: axis = tuple(range(a.ndim))
    else: axis = normalize_axis_tuple(axis,a.ndim)
    if 0 not in axis:
        return np.average(a,axis=axis,weights=weights,returned=returned)
    axis = axis[0]

    a = np.asanyarray(a)

    if weights is None:
        avg = mean_array(a, axis=axis, mpicomm=mpicomm)
        scl = avg.dtype.type(size_array(a)/avg.size)
    else:
        wgt = np.asanyarray(weights)

        if issubclass(a.dtype.type, (np.integer, np.bool_)):
            result_dtype = np.result_type(a.dtype, wgt.dtype, 'f8')
        else:
            result_dtype = np.result_type(a.dtype, wgt.dtype)

        # Sanity checks
        if a.shape != wgt.shape:
            if axis is None:
                raise TypeError(
                    "Axis must be specified when shapes of a and weights "
                    "differ.")
            if wgt.ndim != 1:
                raise TypeError(
                    "1D weights expected when shapes of a and weights differ.")
            if wgt.shape[0] != a.shape[axis]:
                raise ValueError(
                    "Length of weights not compatible with specified axis.")

            # setup wgt to broadcast along axis
            wgt = np.broadcast_to(wgt, (a.ndim-1)*(1,) + wgt.shape)
            wgt = wgt.swapaxes(-1, axis)

        scl = sum_array(wgt, axis=axis, dtype=result_dtype)
        if np.any(scl == 0.0):
            raise ZeroDivisionError(
                "Weights sum to zero, can't be normalized")

        avg = sum_array(np.multiply(a, wgt, dtype=result_dtype), axis=axis)/scl

    if returned:
        if scl.shape != avg.shape:
            scl = np.broadcast_to(scl, avg.shape).copy()
        return avg, scl
    else:
        return avg


@CurrentMPIComm.enable
def var_array(a, axis=-1, fweights=None, aweights=None, ddof=0, mpicomm=None):
    X = np.array(a)
    w = None
    if fweights is not None:
        fweights = np.asarray(fweights, dtype=float)
        if not np.all(fweights == np.around(fweights)):
            raise TypeError(
                "fweights must be integer")
        if fweights.ndim > 1:
            raise RuntimeError(
                "cannot handle multidimensional fweights")
        if fweights.shape[0] != X.shape[axis]:
            raise RuntimeError(
                "incompatible numbers of samples and fweights")
        if any(fweights < 0):
            raise ValueError(
                "fweights cannot be negative")
        w = fweights
    if aweights is not None:
        aweights = np.asarray(aweights, dtype=float)
        if aweights.ndim > 1:
            raise RuntimeError(
                "cannot handle multidimensional aweights")
        if aweights.shape[0] != X.shape[axis]:
            raise RuntimeError(
                "incompatible numbers of samples and aweights")
        if any(aweights < 0):
            raise ValueError(
                "aweights cannot be negative")
        if w is None:
            w = aweights
        else:
            w *= aweights

    avg, w_sum = average_array(X, axis=axis, weights=w, returned=True, mpicomm=mpicomm)

    # Determine the normalization
    if w is None:
        fact = shape_array(a)[axis] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof*sum_array(w*aweights, axis=axis, mpicomm=mpicomm)/w_sum

    if fact <= 0:
        warnings.warn("Degrees of freedom <= 0 for slice",
                      RuntimeWarning, stacklevel=3)
        fact = 0.0

    X = np.apply_along_axis(lambda x: x-avg,axis,X)
    if w is None:
        X_T = X
    else:
        X_T = (X*w)
    c = sum_array(X*X.conj(), axis=axis, mpicomm=mpicomm)
    c *= np.true_divide(1, fact)
    return c.squeeze()


@CurrentMPIComm.enable
def std_array(*args, **kwargs):
    return np.sqrt(var_array(*args,**kwargs))


@CurrentMPIComm.enable
def cov_array(m, y=None, ddof=1, rowvar=True, fweights=None, aweights=None, dtype=None, mpicomm=None):
    # scatter axis is data second axis
    # data (nobs, ndim)
    # Check inputs
    if ddof is not None and ddof != int(ddof):
        raise ValueError("ddof must be integer")

    # Handles complex arrays too
    m = np.asarray(m)
    if m.ndim > 2:
        raise ValueError("m has more than 2 dimensions")

    if y is not None:
        y = np.asarray(y)
        if y.ndim > 2:
            raise ValueError("y has more than 2 dimensions")

    if dtype is None:
        if y is None:
            dtype = np.result_type(m, np.float64)
        else:
            dtype = np.result_type(m, y, np.float64)

    X = np.array(m, ndmin=2, dtype=dtype)
    if not rowvar and X.shape[0] != 1:
        X = X.T
    if X.shape[0] == 0:
        return np.array([]).reshape(0, 0)
    if y is not None:
        y = np.array(y, copy=False, ndmin=2, dtype=dtype)
        if not rowvar and y.shape[0] != 1:
            y = y.T
        X = np.concatenate((X, y), axis=0)

    # Get the product of frequencies and weights
    w = None
    if fweights is not None:
        fweights = np.asarray(fweights, dtype=float)
        if not np.all(fweights == np.around(fweights)):
            raise TypeError(
                "fweights must be integer")
        if fweights.ndim > 1:
            raise RuntimeError(
                "cannot handle multidimensional fweights")
        if fweights.shape[0] != X.shape[1]:
            raise RuntimeError(
                "incompatible numbers of samples and fweights")
        if any(fweights < 0):
            raise ValueError(
                "fweights cannot be negative")
        w = fweights
    if aweights is not None:
        aweights = np.asarray(aweights, dtype=float)
        if aweights.ndim > 1:
            raise RuntimeError(
                "cannot handle multidimensional aweights")
        if aweights.shape[0] != X.shape[1]:
            raise RuntimeError(
                "incompatible numbers of samples and aweights")
        if any(aweights < 0):
            raise ValueError(
                "aweights cannot be negative")
        if w is None:
            w = aweights
        else:
            w *= aweights

    avg, w_sum = average_array(X.T, axis=0, weights=w, returned=True, mpicomm=mpicomm)
    w_sum = w_sum[0]

    # Determine the normalization
    if w is None:
        fact = shape_array(X.T)[0] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof*sum_array(w*aweights, mpicomm=mpicomm)/w_sum

    if fact <= 0:
        warnings.warn("Degrees of freedom <= 0 for slice",
                      RuntimeWarning, stacklevel=3)
        fact = 0.0

    X -= avg[:, None]
    if w is None:
        X_T = X.T
    else:
        X_T = (X*w).T
    c = dot_array(X, X_T.conj(), mpicomm=mpicomm)
    c *= np.true_divide(1, fact)
    return c.squeeze()


@CurrentMPIComm.enable
def corrcoef_array(x, y=None, rowvar=True, fweights=None, aweights=None, dtype=None, mpicomm=None):
    c = cov_array(x, y, rowvar, fweights=None, aweights=None, dtype=dtype, mpicomm=mpicomm)
    try:
        d = np.diag(c)
    except ValueError:
        # scalar covariance
        # nan if incorrect value (nan, inf, 0), 1 otherwise
        return c / c
    stddev = np.sqrt(d.real)
    c /= stddev[:, None]
    c /= stddev[None, :]

    # Clip real and imaginary parts to [-1, 1].  This does not guarantee
    # abs(a[i,j]) <= 1 for complex arrays, but is the best we can do without
    # excessive work.
    np.clip(c.real, -1, 1, out=c.real)
    if np.iscomplexobj(c):
        np.clip(c.imag, -1, 1, out=c.imag)

    return c
