import math
from fractions import Fraction
import logging

import numpy as np
from scipy import special
from cosmoprimo import CorrelationToPower

from cosmopipe.lib.utils import BaseClass


class CorrelationWindowMatrix(BaseClass):

    logger = logging.getLogger('CorrelationWindowMatrix')

    def __init__(self, ells, ellsout=None, window=None, s=None, **kwargs):
        if ellsout is None: ellsout = ells
        self.ells, self.ellsout = ells, ellsout
        self.s = np.asarray(s)
        matrix = np.empty((len(self.ells),len(self.ellsout)) + self.s.shape,dtype=self.s.dtype)
        for illout,ellout in enumerate(self.ellsout):
            for illin,ellin in enumerate(self.ells):
                ellsw,coeffs = wigner3j_square(ellout,ellin)
                matrix[illout][illin] = np.sum([coeff*window(self.s,ell=ell) for ell,coeff in zip(ellsw,coeffs)],axis=0)
        self.matrix = np.transpose(matrix,(2,1,0)) # matrix is now (s,ells,ellsout)

    def compute(self, func):
        # matrix is (ellsout,ells,s), func is (s,ells)
        return np.sum(self.matrix*func,axis=1)

    def __getstate__(self, state):
        for key in ['ells','ellsout','s','matrix']:
            state[key] = getattr(self,key)
        return state


class PowerWindowMatrix(BaseClass):

    logger = logging.getLogger('PowerWindowMatrix')

    def __init__(self, ells, ellsout=None, window=None, kout=None, krange=None, srange=(1e-4,1e4), ns=1024*16, q=1.5, **kwargs):
        if krange is not None:
            srange = (1./krange[1],1./krange[0])
        self.kout = kout
        s = np.logspace(np.log10(srange[0]),np.log10(srange[1]),ns)
        CorrelationWindowMatrix.__init__(self,ells=ells,ellsout=ellsout,window=window,s=s,**kwargs) # matrix is now (s,ells,ellsout)
        ells = np.array(self.ells)
        full_matrix = []

        for illout,ellout in enumerate(self.ellsout):
            matrix = self.matrix[...,illout].T[None,...] * special.spherical_jn(ellout,self.kout[:,None,None]*s) # matrix is now (kout,ells,s)
            #from hankl import P2xi, xi2P
            fflog = CorrelationToPower(s,ell=ells,q=q,lowring=False) # prefactor is 4 pi (-i)^ell
            k, matrix = fflog(matrix) # now (kout,ells,k)
            matrix = np.transpose(matrix,(0,2,1)) # now (kout,k,ells)
            self.k = k[0]
            prefactor = 1./(2.*np.pi**2) * (-1j)**ellout * (-1)**ells # now prefactor 2/pi (-i)^ellout i^ell
            if ellout % 2 == 1: prefactor *= -1j # we provide the imaginary part of odd power spectra, so let's multiply by (-i)^ellout
            prefactor[ells % 2 == 1] *= 1j # we take in the imaginary part of odd power spectra, so let's multiply by i^ell
            matrix = np.real(prefactor * matrix) # everything should be real now
            full_matrix.append(matrix)

        self.matrix = np.transpose(full_matrix,(1,0,2,3)) # now (kout,ellsout,k,ells)

    def compute(self, func):
        # matrix is (kout,ellsout,k,ells), func is (k,ells)
        return np.sum(self.matrix*func,axis=(2,3))

    def __getstate__(self, state):
        for key in ['ells','ellsout','k','kout','matrix']:
            state[key] = getattr(self,key)
        return state


class PowerWindowWideAngleMatrix(BaseClass):

    logger = logging.getLogger('PowerWindowWideAngleMatrix')

    def __init__(self, data, model_bases=None, n=None, **kwargs):
        self.model_bases = ProjectionBaseCollection(model_bases)
        self.data = data

    def compute(self, func):
        # matrix is (kout,ellsout,k,ells), func is (k,ells)
        return np.sum(self.matrix*func,axis=(2,3))

    def __getstate__(self, state):
        for key in ['ells','ellsout','k','kout','matrix']:
            state[key] = getattr(self,key)
        return state



def wigner3j_square(ellout, ellin, prefactor=True, as_string=False):

    coeffs = []
    qvals = []
    retstr = []

    def G(p):
        """Return the function G(p), as defined in Wilson et al 2015.
        See also: WA Al-Salam 1953
        Taken from https://github.com/nickhand/pyRSD.

        Returns
        -------
        numer, denom: int
            the numerator and denominator

        """
        toret = 1
        for p in range(1,p+1): toret *= (2*p - 1)
        return toret, math.factorial(p)

    for p in range(min(ellin,ellout)+1):

        numer, denom = [], []

        # numerator of product of G(x)
        for r in [G(ellout-p), G(p), G(ellin-p)]:
            numer.append(r[0])
            denom.append(r[1])

        # divide by this
        a,b = G(ellin+ellout-p)
        numer.append(b)
        denom.append(a)

        numer.append(2*(ellin+ellout) - 4*p + 1)
        denom.append(2*(ellin+ellout) - 2*p + 1)

        q = ellin + ellout - 2*p
        if prefactor:
            numer.append(2*ellout + 1)
            denom.append(2*q + 1)

        numer = Fraction(np.prod(numer))
        denom = Fraction(np.prod(denom))
        if not as_string:
            coeffs.append(numer*1./denom)
            qvals.append(q)
        else:
            retstr.append('l{:d} {}'.format(q,numer/denom))

    if not as_string:
        return qvals[::-1], coeffs[::-1]
    return retstr[::-1]
