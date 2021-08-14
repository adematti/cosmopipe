"""Global utilities."""

import os
import sys
import time
import functools
import logging
import math
from collections import UserDict

import numpy as np
from numpy.linalg import LinAlgError

from pypescript.utils import setup_logging, mkdir, savefile, snake_to_pascal_case, ScatteredBaseClass, TaskManager, MemoryMonitor, exception_handler
from pypescript.utils import BaseClass as _BaseClass
from pypescript.utils import ScatteredBaseClass as _ScatteredBaseClass
from pypescript import mpi


def dict_nonedefault(d1, **d2):
    """
    Set ``d1`` dictionary entries to those in ``d2`` if do not exist or ``None``.
    Returns updated ``d1``. This is an in-place operation (input ``d1`` is modified).
    """
    for key,value in d2.items():
        if d1.get(key,None) is None:
            d1[key] = value
    return d1


def customspace(min=0., max=1., nbins=None, step=None, scale='linear'):
    """
    Return regularly-spaced array.

    Parameters
    ----------
    min : float, default=0
        Minimum array value.

    max : float, default=1
        Maximum array value.

    nbins : int, default=None
        Number of bins, i.e. array size - 1.
        If ``None``, number of bins determined as nearest integer to ``(max - min)/step``.

    step : int, default=None
        Separation between array values. Used only if ``nbins`` is not ``None``.

    scale : string, default='linear'
        Scaling relation between array values.
        If ``log10`` (or ``log``), ``min``, ``max`` and ``step`` are considered in log10-space.

    Returns
    -------
    toret : array
    """
    if nbins is None:
        nbins = np.rint((max - min)/step).astype(int)
    toret = np.linspace(min,max,nbins+1)
    if scale in ['log','log10']:
        toret = 10**toret
    return toret


class ScatteredBaseClass(_ScatteredBaseClass):

    @classmethod
    @mpi.CurrentMPIComm.enable
    def load(cls, filename, mpiroot=0, mpistate=mpi.CurrentMPIState.GATHERED, mpicomm=None):
        """
        Load class in *numpy* binary format from disk.
        If the loaded state contains ``__class__`` and that exists in ``cls._registry``,
        return instance of ``cls._registry[__class__]`` (instead of ``cls``).
        """
        cls.log_info('Loading {}.'.format(filename),rank=0)
        new = cls.__new__(cls)
        new.mpicomm = mpicomm
        new.mpiroot = mpiroot
        if new.is_mpi_root():
            state = np.load(filename,allow_pickle=True)[()]
            if '__class__' in state:
                clsname = state['__class__']
                if isinstance(clsname,str):
                    if hasattr(cls,'_registry'):
                        cls = cls._registry[clsname]
                else:
                    cls = clsname
            new = cls.from_state(state,mpiroot=mpiroot,mpicomm=mpicomm)
        new.mpistate = mpi.CurrentMPIState.GATHERED
        new = new.mpi_to_state(mpistate)
        return new


class BaseClass(_BaseClass):

    def save_auto(self, *args, **kwargs):
        """If different formats are possible, this method should between them based on file name extension."""
        return self.save(*args,**kwargs)

    def load_auto(self, *args, **kwargs):
        """If different formats are possible, this method should between them based on file name extension."""
        return self.load(*args,**kwargs)

    @classmethod
    @mpi.CurrentMPIComm.enable
    def load(cls, filename, mpiroot=0, mpicomm=None):
        """
        Load class in numpy binary format from disk.
        If the loaded state contains ``__class__`` and that exists in ``cls._registry``,
        return instance of ``cls._registry[__class__]`` (instead of ``cls``).
        """
        cls.log_info('Loading {}.'.format(filename))
        new = cls.__new__(cls)
        new.mpicomm = mpicomm
        new.mpiroot = mpiroot
        if new.is_mpi_root():
            state = np.load(filename,allow_pickle=True)[()]
        state = new.mpicomm.bcast(state if new.is_mpi_root() else None,root=new.mpiroot)
        if '__class__' in state:
            clsname = state['__class__']
            if isinstance(clsname,str):
                if hasattr(cls,'_registry'):
                    try:
                        cls = cls._registry[clsname]
                    except KeyError:
                        pass
            else:
                cls = clsname
        new = cls.from_state(state,mpiroot=mpiroot,mpicomm=mpicomm)
        return new


def _drop_none(di):
    return {key:value for key,value in di.items() if value is not None}


class BaseNameSpace(BaseClass):

    """Class holding a bunch of attributes."""

    _attrs = ['name']

    def set(self, **kwargs):
        """Set attributes ``kwargs``."""
        for name,value in kwargs.items():
            if name in self._attrs:
                setattr(self,name,value)
            else:
                raise ValueError('Cannot set (unknown) attribute {} (available: {})'.format(name,self._attrs))

    def setdefault(self, **kwargs):
        """Set attributes ``kwargs`` if current values are ``None``."""
        for name,value in kwargs.items():
            if name in self._attrs:
                if getattr(self,name,None) is None:
                    setattr(self,name,value)
            else:
                raise ValueError('Cannot set (unknown) attribute {} (available: {})'.format(name,self._attrs))

    def get(self, name, default=None):
        """Return attribute ``name``, defaulting to ``default``."""
        toret = getattr(self,name,default)
        if toret is None:
            return default
        return toret

    def copy(self, **kwargs):
        """Return copy, setting attributes ``kwargs`` on-the-fly."""
        new = self.__copy__()
        new.set(**kwargs)
        return new

    def __repr__(self):
        """String representation including only attributes with non-``None`` values."""
        toret = ['{}={}'.format(name,value) for name,value in self.as_dict(drop_none=True).items()]
        return '{}({})'.format(self.__class__.__name__,','.join(toret))

    def __eq__(self, other):
        """Is ``self`` equal to ``other``, i.e. same type and attributes?"""
        return type(other) == self.__class__ and all(getattr(self,name) == getattr(other,name) for name in self._attrs)

    def eq_ignore_none(self, other):
        """Is ``self`` equal to ``other``, i.e. same type and attributes, ignoring attributes set to ``None``?"""
        return isinstance(other,self.__class__) and all(getattr(self,name) is None or getattr(other,name) is None or getattr(self,name) == getattr(other,name) for name in self._attrs)

    def __hash__(self):
        return hash(self.name)

    def __gt__(self, other):
        raise NotImplementedError

    def __lt__(self, other):
        raise NotImplementedError

    def as_dict(self, drop_none=True):
        """Return dictionary of attributes, droping those set to ``None`` if ``drop_none`` is ``True``."""
        toret = {name:getattr(self,name,None) for name in self._attrs}
        if drop_none:
            return _drop_none(toret)
        return toret

    def __setstate__(self, state):
        """Set the class state dictionary."""
        for name in self._attrs:
            setattr(self,name,state.get(name,None))

    def __getstate__(self):
        """Return this class state dictionary."""
        return self.as_dict(drop_none=True)


class BaseOrderedCollection(BaseClass):
    """
    Class holding an ordered, unique list of items.

    Note
    ----
    When adding a item equal to another already in the collection, the latter will be replaced by the former.
    Insertion order is conserved.
    """
    _copy_if_datablock_copy = True
    _cast = lambda x: x

    def __init__(self, items=None):
        """
        Initialize :class:`BaseOrderedCollection`.

        Parameters
        ----------
        items : list, object, BaseOrderedCollection
            List of elements to add to the collection.
            If not list, interpreted as single element.
            If :class:`BaseOrderedCollection` instance, update :attr:`__dict__`.
        """
        if isinstance(items,self.__class__):
            self.__dict__.update(items.__dict__)
            return
        if items is None:
            items = []
        if not isinstance(items,list):
            items = [items]
        self.data = []
        for item in items:
            self.set(item)

    def index(self, item):
        """Return index of ``item`` in collection."""
        item = self.__class__._cast(item)
        return self.data.index(item)

    def set(self, item):
        """
        Set item in collection.
        If already in collection (following criteria defined in ``item.__eq__``), replace this stored item by the input one.
        Else, append item to collection.
        """
        item = self.__class__._cast(item)
        if item in self: # always set the last one
            self.data[self.data.index(item)] = item
        else:
            self.data.append(item)

    def __getitem__(self, index):
        """Return item by index in collection."""
        return self.data[index]

    @classmethod
    def concatenate(cls, *others):
        """
        Concatenate input collections.
        Unique items only are kept.
        """
        new = cls(others[0])
        for other in others[1:]:
            other = cls(other)
            for item in other.data:
                new.set(item)
        return new

    def extend(self, other):
        """
        Extend collection with ``other``.
        Unique items only are kept.
        """
        new = self.concatenate(self,other)
        self.__dict__.update(new.__dict__)

    def __eq__(self, other):
        """Is ``self`` equal to ``other``, i.e. same type and list of items?"""
        return type(other) == self.__class__ and other.data == self.data

    def __radd__(self, other):
        """Operation corresponding to ``other + self``."""
        if other in [[],0,None]:
            return self.copy()
        return self.__add__(other)

    def __iadd__(self, other):
        """Operation corresponding to ``self += other``."""
        self.extend(other)
        return self

    def __add__(self, other):
        """Addition of two collections is defined as concatenation."""
        return self.concatenate(self,other)

    def unique(self, key):
        """Return list of unique attribute ``key`` of collection items."""
        return np.unique([getattr(item,key) for item in self]).tolist()

    def __repr__(self):
        """String representation."""
        return '{}({})'.format(self.__class__.__name__,repr(self.data))

    def __len__(self):
        """Collection length, i.e. number of items."""
        return len(self.data)

    def __contains__(self, item):
        """Whether collection contains this item."""
        return self.__class__._cast(item) in self.data

    def __iter__(self):
        """Iterator on collection."""
        return iter(self.data)

    def __copy__(self):
        """Return copy, including shallow-copy of internal list of items."""
        new = super(BaseOrderedCollection,self).__copy__()
        new.data = self.data.copy()
        return new

    def select(self, *args, **kwargs):
        """
        Return new collection, after selection of items whose attribute match input values::

            collection.select(name1=value1,name2=value2)

        returns collection of items whose attributes ``name1`` and `name2`` are ``value1``, ``value2``, respectively. Also::

            collection.select(dict(name1=value1,name2=value2),dict(name2=value3))

        returns collection of items whose attributes ``name1`` and `name2`` are ``value1``, ``value2``, respectively or attribute ``name2`` is ``value3``.
        """
        new = self.__class__()
        if args:
            if kwargs:
                raise ValueError('Cannot provide both an expanded dictionary and a list of dictionaries')
            kwargs = args
        if not isinstance(kwargs,tuple):
            kwargs = [kwargs]
        for item in self.data:
            for kw in kwargs:
                if all(getattr(item,key) == value for key,value in kw.items()):
                    new.data.append(item)
        return new

    def __getstate__(self):
        """Return this class state dictionary."""
        return {'data':[item.__getstate__() if hasattr(item,'__getstate__') else item for item in self]}

    def __setstate__(self, state):
        """Set this class state dictionary."""
        self.data = [self.__class__._cast(item) for item in state['data']]

    def clear(self):
        """Empty collection."""
        self.data.clear()


def _check_inv(mat, invmat, rtol=1e-04, atol=1e-05):
    """
    Check input array ``mat`` and ``invmat`` are matrix inverse.
    Raise :class:`LinAlgError` if input product of input arrays ``mat`` and ``invmat`` is not close to identity
    within relative difference ``rtol`` and absolute difference ``atol``.
    """
    tmp = mat.dot(invmat)
    ref = np.diag(np.ones(tmp.shape[0]))
    if not np.allclose(tmp,ref,rtol=rtol,atol=atol):
        raise LinAlgError('Numerically inacurrate inverse matrix, max absolute diff {:.6f}.'.format(np.max(np.abs(tmp-ref))))


def inv(mat, inv=np.linalg.inv, check=True):
    """
    Return inverse of input 2D or 0D (scalar) array ``mat``.

    Parameters
    ----------
    mat : 2D array, scalar
        Input matrix to invert.

    inv : callable, default=np.linalg.inv
        Function that takes in 2D array and returns its inverse.

    check : bool, default=True
        If inversion inaccurate, raise a :class:`LinAlgError` (see :func:`_check_inv`).

    Returns
    -------
    toret : 2D array, scalar
        Inverse of ``mat``.
    """
    mat = np.asarray(mat)
    if mat.ndim == 0:
        return 1./mat
    if check:
        toret = inv(mat)
    else:
        try:
            toret = inv(mat)
        except LinAlgError:
            pass
    if check:
        _check_inv(mat,toret)
    return toret


def blockinv(blocks, inv=np.linalg.inv, check=True):
    """
    Return inverse of input ``blocks`` matrix.

    Parameters
    ----------
    blocks : list of list of arrays
        Input matrix to invert, in the form of blocks, e.g. ``[[A,B],[C,D]]``.

    inv : callable, default=np.linalg.inv
        Function that takes in 2D array and returns its inverse.

    check : bool, default=True
        If inversion inaccurate, raise a :class:`LinAlgError` (see :func:`_check_inv`).

    Returns
    -------
    toret : 2D array
        Inverse of ``blocks`` matrix.
    """
    def _inv(mat):
        if check:
            toret = inv(mat)
        else:
            try:
                toret = inv(mat)
            except LinAlgError:
                pass
        return toret

    A = blocks[0][0]
    if (len(blocks),len(blocks[0])) == (1,1):
        return _inv(A)
    B = np.bmat(blocks[0][1:]).A
    C = np.bmat([b[0].T for b in blocks[1:]]).A.T
    invD = blockinv([b[1:] for b in blocks[1:]],inv=inv)

    def dot(*args):
        return np.linalg.multi_dot(args)

    invShur = _inv(A - dot(B,invD,C))
    toret = np.bmat([[invShur,-dot(invShur,B,invD)],[-dot(invD,C,invShur), invD + dot(invD,C,invShur,B,invD)]]).A
    if check:
        mat = np.bmat(blocks).A
        _check_inv(mat,toret)
    return toret


def interleave(*a):
    """
    Interleave input arrays ``a``, i.e. return array with, in order:
    First elements of all arrays (with size > 0), then second elements of all arrays with size > 1, etc.

    >>> interleave([0,1,2],[3,4])
    [0, 3, 1, 4, 2]
    """
    fill_shape = a[0].shape[1:]
    lens = np.array(list(map(len,a)))
    total_len = sum(lens)
    toret = np.empty(shape=(total_len,)+fill_shape,dtype=a[0].dtype)
    na = len(a)
    slt,sla = [[] for _ in range(na)],[[] for _ in range(na)]
    lastslt = 0
    for le in np.unique(lens):
        ias = np.flatnonzero(lens >= le)
        for iia,ia in enumerate(ias):
            sta = 0 if not sla[ia] else sla[ia][-1].stop
            stt = lastslt + iia
            slt[ia].append(slice(stt,stt+(le-sta-1)*len(ias)+1,len(ias)))
            sla[ia].append(slice(sta,le))
        lastslt = slt[ia][-1].stop

    for ia,a_ in enumerate(a):
        for slt_,sla_ in zip(slt[ia],sla[ia]):
            toret[slt_] = a_[sla_]
    return toret


def cov_to_corrcoef(cov):
    """
    Return correlation matrix corresponding to input covariance matrix ``cov``.
    If ``cov`` is scalar, return 1.
    """
    if np.ndim(cov) == 0:
        return 1.
    stddev = np.sqrt(np.diag(cov).real)
    c = cov/stddev[:,None]/stddev[None,:]
    return c


def weighted_quantile(x, q, weights=None, axis=None, interpolation='lower'):
    """
    Compute the q-th quantile of the weighted data along the specified axis.

    Parameters
    ----------
    a : array
        Input array or object that can be converted to an array.

    q : tuple, list, array
        Quantile or sequence of quantiles to compute, which must be between
        0 and 1 inclusive.

    weights : array, default=None
        An array of weights associated with the values in ``a``. Each value in
        ``a`` contributes to the cumulative distribution according to its associated weight.
        The weights array can either be 1D (in which case its length must be
        the size of ``a`` along the given axis) or of the same shape as ``a``.
        If ``weights=None``, then all data in ``a`` are assumed to have a
        weight equal to one.
        The only constraint on ``weights`` is that ``sum(weights)`` must not be 0.

    axis : int, tuple, default=None
        Axis or axes along which the quantiles are computed. The
        default is to compute the quantile(s) along a flattened
        version of the array.

    interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}, default='linear'
        This optional parameter specifies the interpolation method to
        use when the desired quantile lies between two data points
        ``i < j``:

        * linear: ``i + (j - i) * fraction``, where ``fraction``
          is the fractional part of the index surrounded by ``i``
          and ``j``.
        * lower: ``i``.
        * higher: ``j``.
        * nearest: ``i`` or ``j``, whichever is nearest.
        * midpoint: ``(i + j) / 2``.

    Returns
    -------
    quantile : scalar, array
        If ``q`` is a single quantile and ``axis=None``, then the result
        is a scalar. If multiple quantiles are given, first axis of
        the result corresponds to the quantiles. The other axes are
        the axes that remain after the reduction of ``a``. If the input
        contains integers or floats smaller than ``float64``, the output
        data-type is ``float64``. Otherwise, the output data-type is the
        same as that of the input. If ``out`` is specified, that array is
        returned instead.

    Note
    ----
    Inspired from https://github.com/minaskar/cronus/blob/master/cronus/plot.py.
    """
    if weights is None:
        # If no weights provided, this simply calls `np.percentile`.
        return np.quantile(x,q,axis=axis,interpolation=interpolation)

    # Initial check.
    x = np.atleast_1d(x)
    isscalar = np.ndim(q) == 0
    q = np.atleast_1d(q)

    # Quantile check.
    if np.any(q < 0.) or np.any(q > 1.):
        raise ValueError('Quantiles must be between 0. and 1.')

    if axis is None:
        axis = range(x.ndim)

    if np.ndim(axis) == 0:
        axis = (axis,)

    if weights.ndim > 1:
        if x.shape != weights.shape:
            raise ValueError('Dimension mismatch: shape(weights) != shape(x).')

    x = np.moveaxis(x,axis,range(x.ndim-len(axis),x.ndim))
    x = x.reshape(x.shape[:-len(axis)]+(-1,))
    if weights.ndim > 1:
        weights = np.moveaxis(weights,axis,range(x.ndim-len(axis),x.ndim))
        weights = weights.reshape(weights.shape[:-len(axis)]+(-1,))
    else:
        reps = x.shape[:-1]+(1,)
        weights = np.tile(weights,reps)

    idx = np.argsort(x,axis=-1) # sort samples
    x = np.take_along_axis(x,idx,axis=-1)
    sw = np.take_along_axis(weights,idx,axis=-1) # sort weights
    cdf = np.cumsum(sw,axis=-1) # compute CDF
    cdf = cdf[...,:-1]
    cdf = cdf/cdf[...,-1][...,None]  # normalize CDF
    zeros = np.zeros_like(cdf,shape=cdf.shape[:-1]+(1,))
    cdf = np.concatenate([zeros,cdf],axis=-1)  # ensure proper span
    idx0 = np.apply_along_axis(np.searchsorted,-1,cdf,q,side='right') - 1
    if interpolation != 'higher':
        q0 = np.take_along_axis(x,idx0,axis=-1)
    if interpolation != 'lower':
        idx1 = np.clip(idx0+1,None,x.shape[-1]-1)
        q1 = np.take_along_axis(x,idx1,axis=-1)
    if interpolation in ['nearest','linear']:
        cdf0,cdf1 = np.take_along_axis(cdf,idx0,axis=-1),np.take_along_axis(cdf,idx1,axis=-1)
    if interpolation == 'nearest':
        mask_lower = q - cdf0 < cdf1 - q
        quantiles = q1
        # in place, q1 not used in the following
        quantiles[mask_lower] = q0[mask_lower]
    if interpolation == 'linear':
        step = cdf1 - cdf0
        diff = q - cdf0
        mask = idx1 == idx0
        step[mask] = diff[mask]
        fraction = diff/step
        quantiles = q0 + fraction * (q1 - q0)
    if interpolation == 'lower':
        quantiles = q0
    if interpolation == 'higher':
        quantiles = q1
    if interpolation == 'midpoint':
        quantiles = (q0 + q1)/2.
    quantiles = quantiles.swapaxes(-1,0)
    if isscalar:
        return quantiles[0]
    return quantiles


def rebin_ndarray(ndarray, new_edges, statistic=np.sum, interpolation='linear'):
    """
    Bin an ndarray in all axes based on the target shape, by summing or
    averaging. Number of output dimensions must match number of input dimensions and
    new axes must divide old ones.

    Taken from https://stackoverflow.com/questions/8090229/resize-with-averaging-or-rebin-a-numpy-2d-array
    and https://nbodykit.readthedocs.io/en/latest/_modules/nbodykit/binned_statistic.html#BinnedStatistic.reindex.

    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)

    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]

    """
    from scipy import stats

    def hist_laxis(data, edges, statistic='sum'):
        # Setup bins and determine the bin location for each element for the bins
        n = data.shape[-1]
        data_2d = data.reshape(-1,n)
        idx = np.searchsorted(edges,np.arange(data_2d.shape[-1]),'right') - 1
        idx = np.tile(idx,(data.shape[0],1))
        nbins = len(edges)-1

        # Some elements would be off limits, so get a mask for those
        mask = (idx==-1) | (idx==nbins)

        # We need to use bincount to get bin based counts. To have unique IDs for
        # each row and not get confused by the ones from other rows, we need to
        # offset each row by a scale (using row length for this).
        scaled_idx = nbins*np.arange(data_2d.shape[0])[:,None] + idx

        # Set the bad ones to be last possible index+1 : n_bins*data2D.shape[0]
        limit = nbins*data_2d.shape[0] + 1
        scaled_idx[mask] = limit

        # Get the counts and reshape to multi-dim
        #counts = np.bincount(scaled_idx.ravel(),minlength=limit+1)[:-1]
        bins = np.arange(0,limit)
        counts = stats.binned_statistic(scaled_idx.ravel(),values=data_2d.ravel(),statistic=statistic,bins=bins)[0]
        counts.shape = data.shape[:-1] + (nbins,)
        return counts

    if ndarray.ndim != len(new_edges):
        raise ValueError('Input array dim is {}, but requested output one is {}'.format(ndarray.ndim,len(new_edges)))
    _new_edges = new_edges
    new_edges = []
    for s,e in zip(ndarray.shape,_new_edges):
        if np.ndim(e) == 0:
            if s % e != 0:
                raise ValueError('If int, new edges must be a divider of the original shape')
            e = np.arange(0,s+1,s//e)
        new_edges.append(e)

    for i,e in enumerate(new_edges):
        ndarray = hist_laxis(ndarray.swapaxes(i,-1),e,statistic).swapaxes(-1,i)

    if interpolation:
        from scipy import interpolate
        for i,e in enumerate(new_edges):
            ie = np.floor(e)
            ie = (ie[1:] + ie[:-1])/2.
            e = (e[1:] + e[:-1])/2.
            if ie.size > 1:
                ndarray = interpolate.interp1d(ie,ndarray,kind=interpolation,axis=i,copy=True,bounds_error=None,fill_value='extrapolate',assume_sorted=True)(e)

    return ndarray


def enforce_shape(x, y, grid=True):
    """
    Broadcast ``x`` and ``y`` arrays.

    Parameters
    ----------
    x : array_like
        Input x-array, scalar, 1D or 2D if not ``grid``.

    y : array_like
        Input y-array, scalar, 1D or 2D if not ``grid``.

    grid : bool, default=True
        If ``grid``, and ``x``, ``y`` not scalars, add dimension to ``x`` such that ``x`` and ``y`` can be broadcast
        (e.g. ``x*y``, is shape ``(len(x),) + y.shape``).
        Else, simply return x,y.

    Returns
    -------
    x : array
        x-array.

    y : array
        y-array.
    """
    x,y = np.asarray(x),np.asarray(y)
    if (not grid) or (x.ndim == 0) or (y.ndim == 0):
        return x,y
    return x[:,None],y


def txt_to_latex(txt):
    """Transform standard text into latex by replacing '_xxx' with '_{xxx}' and '^xxx' with '^{xxx}'."""
    latex = ''
    txt = list(txt)
    for c in txt:
        latex += c
        if c in ['_','^']:
            latex += '{'
            txt += '}'
    return latex


def std_notation(value, sigfigs, extra=None):
    """
    Standard notation (US version).
    Return a string corresponding to value with the number of significant digits ``sigfigs``.

    >>> std_notation(5, 2)
    '5.0'
    >>> std_notation(5.36, 2)
    '5.4'
    >>> std_notation(5360, 2)
    '5400'
    >>> std_notation(0.05363, 3)
    '0.0536'

    Created by William Rusnack:
      github.com/BebeSparkelSparkel
      linkedin.com/in/williamrusnack/
      williamrusnack@gmail.com
    """
    sig_digits, power, is_neg = _number_profile(value, sigfigs)
    if is_neg and all(d == '0' for d in sig_digits): is_neg = False

    return ('-' if is_neg else '') + _place_dot(sig_digits, power)


def sci_notation(value, sigfigs, filler='e'):
    """
    Scientific notation.

    Return a string corresponding to value with the number of significant digits ``sigfigs``,
    with 10s exponent filler ``filler`` placed between the decimal value and 10s exponent.

    >>> sci_notation(123, 1, 'e')
    '1e2'
    >>> sci_notation(123, 3, 'e')
    '1.23e2'
    >>> sci_notation(0.126, 2, 'e')
    '1.3e-1'

    Created by William Rusnack
      github.com/BebeSparkelSparkel
      linkedin.com/in/williamrusnack/
      williamrusnack@gmail.com
    """
    sig_digits, power, is_neg = _number_profile(value, sigfigs)
    if is_neg and all(d == '0' for d in sig_digits): is_neg = False

    dot_power = min(-(sigfigs - 1),0)
    ten_power = power + sigfigs - 1
    return ('-' if is_neg else '') + _place_dot(sig_digits, dot_power) + filler + str(ten_power)


def _place_dot(digits, power):
    """
    Place dot in the correct spot, given by integer ``power`` (starting from the right of ``digits``)
    in the string ``digits``.
    If the dot is outside the range of the digits zeros will be added.

    >>> _place_dot('123', 2)
    '12300'
    >>> _place_dot('123', -2)
    '1.23'
    >>> _place_dot('123', 3)
    '0.123'
    >>> _place_dot('123', 5)
    '0.00123'

    Created by William Rusnack
      github.com/BebeSparkelSparkel
      linkedin.com/in/williamrusnack/
      williamrusnack@gmail.com
    """
    if power > 0: out = digits + '0' * power

    elif power < 0:
        power = abs(power)
        sigfigs = len(digits)

        if power < sigfigs:
            out = digits[:-power] + '.' + digits[-power:]

        else:
            out = '0.' + '0' * (power - sigfigs) + digits

    else:
        out = digits + ('.' if digits[-1] == '0' else '')

    return out


def _number_profile(value, sigfigs):
    """
    Return elements to turn number into string representation.

    Created by William Rusnack
      github.com/BebeSparkelSparkel
      linkedin.com/in/williamrusnack/
      williamrusnack@gmail.com

    Parameters
    ----------
    value : float
        Number.

    sigfigs : int
        Number of significant digits.

    Returns
    -------
    sig_digits : string
        Significant digits.

    power : int
        10s exponent to get the dot to the proper location in the significant digits

    is_neg : bool
        ``True`` if value is < 0 else ``False``
    """
    if value == 0:
        sig_digits = '0' * sigfigs
        power = -(1 - sigfigs)
        is_neg = False

    else:
        if value < 0:
            value = abs(value)
            is_neg = True
        else:
            is_neg = False

        power = -1 * math.floor(math.log10(value)) + sigfigs - 1
        sig_digits = str(int(round(abs(value) * 10.0**power)))

    return sig_digits, int(-power), is_neg


def round_measurement(x, u=0.1, v=None, sigfigs=2, notation='auto'):
    """
    Return string representation of input central value ``x`` with uncertainties ``u`` and ``v``.

    Parameters
    ----------
    x : float
        Central value.

    u : float, default=0.1
        Upper uncertainty on ``x`` (positive).

    v : float, default=None
        Lower uncertainty on ``v`` (negative).
        If ``None``, only returns string representation for ``x`` and ``u``.

    sigfigs : int, default=2
        Number of digits to keep for the uncertainties (hence fixing number of digits for ``x``).

    Returns
    -------
    xr : string
        String representation for central value ``x``.

    ur : string
        String representation for upper uncertainty ``u``.

    vr : string
        If ``v`` is not ``None``, string representation for lower uncertainty ``v``.
    """
    x,u = float(x),float(u)
    return_v = True
    if v is None:
        return_v = False
        v = -abs(u)
    else:
        v = float(v)
    logx = 0
    if x != 0.: logx = math.floor(math.log10(abs(x)))
    if u == 0.: logu = logx
    else: logu = math.floor(math.log10(abs(u)))
    if v == 0.: logv = logx
    else: logv = math.floor(math.log10(abs(v)))
    if x == 0.: logx = max(logu,logv)

    def round_notation(val, sigfigs, notation='auto', center=False):
        if notation == 'auto':
            #if 1e-3 < abs(val) < 1e3 or center and (1e-3 - abs(u) < abs(x) < 1e3 + abs(v)):
            if (1e-3 - abs(u) < abs(x) < 1e3 + abs(v)):
                notation = 'std'
            else:
                notation = 'sci'
        notation_dict = {'std':std_notation,'sci':sci_notation}
        if notation in notation_dict:
            return notation_dict[notation](val,sigfigs=sigfigs)
        return notation(val,sigfigs=sigfigs)

    if logv>logu:
        xr = round_notation(x,sigfigs=logx-logu+sigfigs,notation=notation,center=True)
        ur = round_notation(u,sigfigs=sigfigs,notation=notation)
        vr = round_notation(v,sigfigs=logv-logu+sigfigs,notation=notation)
    else:
        xr = round_notation(x,sigfigs=logx-logv+sigfigs,notation=notation,center=True)
        ur = round_notation(u,sigfigs=logu-logv+sigfigs,notation=notation)
        vr = round_notation(v,sigfigs=sigfigs,notation=notation)

    if return_v: return xr, ur, vr
    return xr, ur

def match1d(id1, id2):
    """
    Match ``id2`` array to ``id1`` array.

    Parameters
    ----------
    id1 : array
        IDs 1, should be unique.

    id2 : array
        IDs 2, should be unique.

    Returns
    -------
    index1 : array
        Indices of matching ``id1``.

    index2 : array
        Indices of matching ``id2``.

    Warning
    -------
    Makes sense only if ``id1`` and ``id2`` elements are unique.

    References
    ----------
    https://www.followthesheep.com/?p=1366
    """
    sort1 = np.argsort(id1)
    sort2 = np.argsort(id2)
    sortleft1 = id1[sort1].searchsorted(id2[sort2],side='left')
    sortright1 = id1[sort1].searchsorted(id2[sort2],side='right')
    sortleft2 = id2[sort2].searchsorted(id1[sort1],side='left')
    sortright2 = id2[sort2].searchsorted(id1[sort1],side='right')

    ind2 = np.flatnonzero(sortright1-sortleft1 > 0)
    ind1 = np.flatnonzero(sortright2-sortleft2 > 0)

    return sort1[ind1], sort2[ind2]
