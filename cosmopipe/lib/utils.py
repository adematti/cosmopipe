import os
import sys
import time
import functools
import logging
import math
from collections import UserDict

import numpy as np
from numpy.linalg import LinAlgError

from pypescript.utils import setup_logging, mkdir, savefile, snake_to_pascal_case, BaseClass, ScatteredBaseClass, TaskManager, MemoryMonitor
from pypescript.utils import split_section_name
from pypescript.config import parse_yaml


class OrderedMapping(BaseClass,UserDict):

    def __init__(self, d=None, order=None):
        self.data = d or {}
        if order is not None and not callable(order):

            def order_(key):
                try:
                    return order.index(key)
                except ValueError:
                    return -1

            self.order = order_
        else:
            self.order = order

    def keys(self):
        """Return keys sorted by chronological order in :mod:`legacypipe.runbrick`."""
        return sorted(self.data.keys(),key=self.order)

    def __iter__(self):
        """Iterate."""
        return iter(self.keys())


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
            for key in mapping.keys():
                keys.append(key)
                self.array[array.astype(type(key)) == key] = keys.index(key)
            self.keys = keys
        except AttributeError:
            self.array = np.array(array,dtype=dtype)

    def __eq__(self,other):
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
        raise LinalgError('Numerically inacurrate inverse matrix, max absolute diff {:.3f}.'.format(np.max(np.abs(tmp-ref))))


def inv(mat, inv=np.linalg.inv, check=True):
    toret = inv(mat)
    if check:
        _check_inv(mat,toret)
    return toret


def blockinv(blocks, inv=np.linalg.inv, check=True):
    A = blocks[0][0]
    if (len(blocks),len(blocks[0])) == (1,1):
        return inv(A)
    B = np.bmat(blocks[0][1:]).A
    C = np.bmat([b[0].T for b in blocks[1:]]).A.T
    invD = blockinv([b[1:] for b in blocks[1:]],inv=inv)

    def dot(*args):
        return np.linalg.multi_dot(args)

    invShur = inv(A - dot(B,invD,C))
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
    d = np.diag(cov)
    stddev = np.sqrt(d.real)
    c = cov/stddev[:, None]
    c /= stddev[None, :]
    return c


def weighted_quantile(x, q, weights=None):
    """
    Compute (weighted) quantiles from an input set of samples.
    Taken from https://github.com/minaskar/cronus/blob/master/cronus/plot.py.

    Parameters
    ----------
    x : `~numpy.ndarray` with shape (nsamps,)
        Input samples.
    q : `~numpy.ndarray` with shape (nquantiles,)
       The list of quantiles to compute from `[0., 1.]`.
    weights : `~numpy.ndarray` with shape (nsamps,), optional
        The associated weight from each sample.

    Returns
    -------
    quantiles : `~numpy.ndarray` with shape (nquantiles,)
        The weighted sample quantiles computed at `q`.
    """

    # Initial check.
    x = np.atleast_1d(x)
    q = np.atleast_1d(q)

    # Quantile check.
    if np.any(q < 0.) or np.any(q > 1.):
        raise ValueError("Quantiles must be between 0. and 1.")

    if weights is None:
        # If no weights provided, this simply calls `np.percentile`.
        return np.quantile(x, q)
    else:
        # If weights are provided, compute the weighted quantiles.
        weights = np.atleast_1d(weights)
        if len(x) != len(weights):
            raise ValueError("Dimension mismatch: len(weights) != len(x).")
        idx = np.argsort(x)  # sort samples
        sw = weights[idx]  # sort weights
        cdf = np.cumsum(sw)[:-1]  # compute CDF
        cdf /= cdf[-1]  # normalize CDF
        cdf = np.append(0, cdf)  # ensure proper span
        quantiles = np.interp(q, cdf, x[idx])
        return quantiles


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
    if x != 0.: logx = -math.floor(math.log10(abs(x)))
    if u == 0.: logu = logx
    else: logu = -math.floor(math.log10(abs(u)))
    if v == 0.: logv = logx
    else: logv = -math.floor(math.log10(abs(v)))
    if x == 0.: logx = min(logu,logv)

    def round_notation(value, sigfigs, notation='auto'):
        if notation == 'auto':
            if 1e-3 - abs(u) < abs(x) < 1e3 + v:
                notation = 'std'
            else:
                notation = 'sci'
        notation_dict = {'std':std_notation,'sci':sci_notation}
        if notation in notation_dict:
            return notation_dict[notation](value,sigfigs=sigfigs)
        return notation(value,sigfigs=sigfigs)

    if logv>logu:
        xr = round_notation(x,sigfigs=logv-logx+sigfigs,notation=notation)
        ur = round_notation(u,sigfigs=logv-logu+sigfigs,notation=notation)
        vr = round_notation(v,sigfigs=sigfigs,notation=notation)
    else:
        xr = round_notation(x,sigfigs=logu-logx+sigfigs,notation=notation)
        ur = round_notation(u,sigfigs=sigfigs,notation=notation)
        vr = round_notation(v,sigfigs=logu-logv+sigfigs,notation=notation)

    if return_v: return xr, ur, vr
    return xr, ur


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

    def round_notation(value, sigfigs, notation='auto', center=True):
        if notation == 'auto':
            if 1e-3 < abs(value) < 1e3 or center and (1e-3 - abs(u) < abs(value) < 1e3 + v):
                notation = 'std'
            else:
                notation = 'sci'
        notation_dict = {'std':std_notation,'sci':sci_notation}
        if notation in notation_dict:
            return notation_dict[notation](value,sigfigs=sigfigs)
        return notation(value,sigfigs=sigfigs)

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
