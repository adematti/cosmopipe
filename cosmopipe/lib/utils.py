import os
import sys
import time
import functools
import logging
import math
from collections import UserDict

import numpy as np
from numpy.linalg import LinAlgError

from pypescript.utils import setup_logging, mkdir, savefile, snake_to_pascal_case, ScatteredBaseClass, TaskManager, MemoryMonitor
from pypescript.utils import BaseClass as _BaseClass


class BaseClass(_BaseClass):

    def save_auto(self, *args, **kwargs):
        return self.save(*args,**kwargs)

    def load_auto(self, *args, **kwargs):
        return self.load(*args,**kwargs)


class MappingArray(BaseClass):

    def __init__(self, array, mapping=None, dtype=None):

        if isinstance(array,self.__class__):
            self.__dict__.update(array.__dict__)
            return

        if mapping is None:
            mapping = np.unique(array).tolist()
            mapping = {m:m for m in mapping}

        self.keys = mapping
        if dtype is None:
            nbytes = 2**np.ceil(np.log2(np.ceil((np.log2(len(self.keys)+1) + 1.)/8.)))
            dtype = 'i{:d}'.format(int(nbytes))

        try:
            self.array = - np.ones_like(array,dtype=dtype)
            array = np.array(array)
            keys = []
            for key,value in mapping.items():
                keys.append(value)
                self.array[array.astype(type(key)) == key] = keys.index(value)
            self.keys = keys
        except AttributeError:
            self.array = np.array(array,dtype=dtype)

    def __eq__(self, other):
        if other in self.keys:
            return self.array == self.keys.index(other)
        return self.array == other

    def __getitem__(self, name):
        try:
            return self.keys[self.array[name]]
        except TypeError:
            new = self.copy()
            new.keys = self.keys.copy()
            new.array = self.array[name]
            return new

    def __setitem__(self, name, item):
        self.array[name] = self.keys.index(item)

    @property
    def shape(self):
        return self.array.shape

    @property
    def size(self):
        return self.array.size

    def asarray(self):
        return np.array([self.keys[a] for a in self.array.flat]).reshape(self.shape)

    def __getstate__(self):
        state = super(MappingArray,self).__getstate__()
        for key in ['array','keys']:
            state[key] = getattr(self,key)
        return state


def _check_inv(mat, invmat, **kwargs):
    tmp = mat.dot(invmat)
    ref = np.diag(np.ones(tmp.shape[0]))
    if not np.allclose(tmp,ref,**kwargs):
        raise LinAlgError('Numerically inacurrate inverse matrix, max absolute diff {:.3f}.'.format(np.max(np.abs(tmp-ref))))


def inv(mat, inv=np.linalg.inv, check=True):
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
    a : array_like
        Input array or object that can be converted to an array.
    q : array_like of float
        Quantile or sequence of quantiles to compute, which must be between
        0 and 1 inclusive.
    weights : array_like, optional
        An array of weights associated with the values in `a`. Each value in
        `a` contributes to the average according to its associated weight.
        The weights array can either be 1-D (in which case its length must be
        the size of `a` along the given axis) or of the same shape as `a`.
        If `weights=None`, then all data in `a` are assumed to have a
        weight equal to one.  The 1-D calculation is::
            avg = sum(a * weights) / sum(weights)
        The only constraint on `weights` is that `sum(weights)` must not be 0.
    axis : {int, tuple of int, None}, optional
        Axis or axes along which the quantiles are computed. The
        default is to compute the quantile(s) along a flattened
        version of the array.
    interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
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
    quantile : scalar or ndarray
        If `q` is a single quantile and `axis=None`, then the result
        is a scalar. If multiple quantiles are given, first axis of
        the result corresponds to the quantiles. The other axes are
        the axes that remain after the reduction of `a`. If the input
        contains integers or floats smaller than ``float64``, the output
        data-type is ``float64``. Otherwise, the output data-type is the
        same as that of the input. If `out` is specified, that array is
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
    """Bin an ndarray in all axes based on the target shape, by summing or
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
    standard notation (US version)
    ref: http://www.mathsisfun.com/definitions/standard-notation.html

    returns a string of value with the proper sigfigs

    ex:
      std_notation(5, 2) => 5.0
      std_notation(5.36, 2) => 5.4
      std_notation(5360, 2) => 5400
      std_notation(0.05363, 3) => 0.0536

      created by William Rusnack
        github.com/BebeSparkelSparkel
        linkedin.com/in/williamrusnack/
        williamrusnack@gmail.com
    """
    sig_digits, power, is_neg = _number_profile(value, sigfigs)
    if is_neg and all(d == '0' for d in sig_digits): is_neg = False

    return ('-' if is_neg else '') + _place_dot(sig_digits, power)


def sci_notation(value, sigfigs, filler='e'):
    """
    scientific notation
    ref: https://www.mathsisfun.com/numbers/scientific-notation.html

    returns a string of value with the proper sigfigs and 10s exponent
    filler is placed between the decimal value and 10s exponent

    ex:
      sci_notation(123, 1, 'E') => 1E2
      sci_notation(123, 3, 'E') => 1.23E2
      sci_notation(.126, 2, 'E') => 1.3E-1

      created by William Rusnack
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
    places the dot in the correct spot in the digits
    if the dot is outside the range of the digits zeros will be added

    ex:
      _place_dot(123, 2) => 12300
      _place_dot(123, -2) => 1.23
      _place_dot(123, 3) => 0.123
      _place_dot(123, 5) => 0.00123

      created by William Rusnack
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
    returns:
      string of significant digits
      10s exponent to get the dot to the proper location in the significant digits
      bool that's true if value is less than zero else false

      created by William Rusnack
        github.com/BebeSparkelSparkel
        linkedin.com/in/williamrusnack/
        williamrusnack@gmail.com
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
    x,u = float(x),float(u)
    return_v = True
    if v is None:
        return_v = False
        v = u
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
