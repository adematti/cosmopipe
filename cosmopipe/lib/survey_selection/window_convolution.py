"""Implementation of window function effect."""

import math
from fractions import Fraction
import logging

import numpy as np
from scipy import special
from cosmoprimo import CorrelationToPower

from cosmopipe.lib.utils import BaseClass
from cosmopipe.lib.theory import ProjectionBasis
from cosmopipe.lib.theory.integration import weights_trapz
from cosmopipe.lib.data_vector import ProjectionNameCollection
from .base import BaseRegularMatrix, BaseMatrix


class CorrelationWindowMatrix(BaseRegularMatrix):
    """
    Class computing matrix for window product in configuration space.

    Attributes
    ----------
    projmatrix : array
        Array of shape ``(len(self.projsout),len(self.projsin),len(self.x))``
        to convert input array from one basis to another (e.g. multipoles to wedges).
    """
    logger = logging.getLogger('CorrelationWindowMatrix')
    basis = ProjectionBasis(space=ProjectionBasis.CORRELATION,mode=ProjectionBasis.MULTIPOLE)

    def __init__(self, window=None, sum_wa=True, default_zero=False):
        """
        Initialize :class:`CorrelationWindowMatrix`.

        Parameters
        ----------
        window : WindowFunction
            Window function to multiply correlation function with.

        sum_wa : bool, default=True
            Whether to perform summation over wide-angle orders.

        default_zero : bool, default=False
            If a given projection is not provided in window function, set to 0.
            Else an :class:`IndexError` is raised.
        """
        self.window = window
        self.sum_wa = sum_wa
        self.default_zero = default_zero

    @property
    def s(self):
        """x-coordinates are s-separations."""
        return self.x

    def setup(self, s, projsin, projsout=None):
        """
        Set up transform.

        Parameters
        ----------
        s : array
            Input (and ouput) separations.

        projsin : list, ProjectionNameCollection
            Input projections.

        projsout : list, ProjectionNameCollection, default=None
            Output projections. Defaults to ``propose_out(projsin)[-1]``.
        """
        self.projsin = projsin
        self.projsout = projsout
        if projsout is None:
            self.projsout = self.propose_out(projsin)
        ellsin = [proj.proj for proj in self.projsin]
        ellsout = [proj.proj for proj in self.projsout]
        self.x = np.asarray(s)

        matrix = []
        for projout in self.projsout:
            line = []
            for projin in self.projsin:
                tmp = np.zeros_like(self.x)
                if not self.sum_wa and sum(proj.wa_order is None for proj in [projin,projout]) == 1:
                    raise ValueError('Input and output projections should both have wide-angle order wa_order specified')
                sum_wa = self.sum_wa and projout.wa_order is None
                if sum_wa or projout.wa_order == projin.wa_order:
                    ellsw,coeffs = wigner3j_square(projout.proj,projin.proj)
                    for ell,coeff in zip(ellsw,coeffs):
                        proj = projin.copy(space=ProjectionBasis.CORRELATION,mode=ProjectionBasis.MULTIPOLE,proj=ell)
                        tmp += coeff*self.window(proj,self.s,default_zero=self.default_zero)
                line.append(tmp)
            matrix.append(line)
        self.projmatrix = np.array(matrix)

    @property
    def matrix(self):
        """
        Return 2D array of shape ``(len(self.projsout)*len(self.x),len(self.projsin)*len(self.x))``
        corresponding to :attr:`projmatrix`.
        """
        return np.bmat([[np.diag(tmp) for tmp in line] for line in self.projmatrix]).A

    def propose_out(self, projsin):
        """Propose input and output projection names given proposed input projection names ``projsin``."""
        projsin, projsout = super(PowerWindowMatrix,self).propose_out(projsin)
        if self.sum_wa:
            projsin = ProjectionNameCollection([proj for proj in projsout if proj.wa_order is not None])
            projsout = ProjectionNameCollection([proj.copy(wa_order=None) for proj in projsout])
        return projsin, projsout


class PowerWindowMatrix(BaseMatrix):

    """Class computing matrix for window convolution in Fourier space."""

    logger = logging.getLogger('PowerWindowMatrix')
    basis = ProjectionBasis(space=ProjectionBasis.POWER,mode=ProjectionBasis.MULTIPOLE)
    regularin = True

    def __init__(self, window=None, srange=(1e-4,1e4), krange=None, ns=1024*16, q=0, krebin=1, **kwargs):
        """
        Initialize :class:`PowerWindowMatrix`.

        Parameters
        ----------
        window : WindowFunction
            Window function to convolve power spectrum with.

        srange : tuple, default=(1e-4,1e4)
            :math:`s`-range for Hankel transforms.

        krange : tuple, default=None
            If ``srange`` not provided, equivalent :math:`k`-range (taking :math:`s = 1/k`).

        ns : int, default=1024*16
             Number of log-spaced points for Hankel transforms.

        q : int, default=0
            Power-law tilt to regularize Hankel transforms.

        krebin : int, default=1
            Rebin matrix aling input :math:`k` by factor ``krebin``.

        kwargs : dict
            Arguments for :class:`CorrelationWindowMatrix`.
        """
        if krange is not None:
            srange = (1./krange[1],1./krange[0])
        self.s = np.logspace(np.log10(srange[0]),np.log10(srange[1]),ns)
        self.krange = krange
        self.krebin = krebin
        self.q = q
        CorrelationWindowMatrix.__init__(self,window=window,**kwargs)

    @property
    def kout(self):
        """Input x-coordinates are k-wavenumbers."""
        return self.xout

    @property
    def kin(self):
        """Output x-coordinates are k-wavenumbers."""
        return self.xin

    def setup(self, kout, projsin, projsout=None):
        """
        Set up transform.

        Parameters
        ----------
        kout : array
            Output wavenumbers.

        projsin : list, ProjectionNameCollection
            Input projections.

        projsout : list, ProjectionNameCollection, default=None
            Output projections. Defaults to ``propose_out(projsin)[-1]``.
        """
        self.projsin = ProjectionNameCollection(projsin)
        if projsout is None:
            self.projsout = self.propose_out(projsin)[-1]
        else:
            self.projsout = ProjectionNameCollection(projsout)
        self.xout = kout
        if np.ndim(kout[0]) == 0:
            self.xout = [kout]*len(projsout)
        self.xout = [np.asarray(x) for x in self.xout]
        CorrelationWindowMatrix.setup(self,self.s,projsin=self.projsin,projsout=self.projsout)
        self.corrmatrix = self.projmatrix

        matrix = []
        for iout,projout in enumerate(self.projsout):
            line = []
            for iin,projin in enumerate(self.projsin):
                tmp = special.spherical_jn(projout.proj,self.kout[iout][:,None]*self.s) * self.corrmatrix[iout,iin] # matrix is now (kout,s)
                #from hankl import P2xi, xi2P
                fflog = CorrelationToPower(self.s,ell=projin.proj,q=self.q,lowring=False) # prefactor is 4 pi (-i)^ellin
                self.xin, tmp = fflog(tmp) # now (kout,k)
                prefactor = 1./(2.*np.pi**2) * (-1j)**projout.proj * (-1)**projin.proj # now prefactor 2/pi (-i)^ellout i^ellin
                if projout.proj % 2 == 1: prefactor *= -1j # we provide the imaginary part of odd power spectra, so let's multiply by (-i)^ellout
                if projin.proj % 2 == 1: prefactor *= 1j # we take in the imaginary part of odd power spectra, so let's multiply by i^ellin
                tmp = np.real(prefactor * tmp) * weights_trapz(self.kin**3) / 3. # everything should be real now
                if self.krebin > 1:
                    from scipy import signal
                    tmp = signal.convolve(tmp,np.ones((1,self.krebin)),mode='valid') / self.krebin
                    self.xin = signal.convolve(self.xin,np.ones(self.krebin),mode='valid') / self.krebin
                if self.krange is not None:
                    mask = (self.xin >= self.krange[0]) & (self.xin <= self.krange[-1])
                    self.xin = self.xin[mask]
                    tmp = tmp[:,mask]
                #print(projout.proj,projin.proj,self.xin.min(),self.xin.max(),self.kout[iout].min(),self.kout[iout].max(),tmp.min(),tmp.max())
                line.append(tmp)
            matrix.append(line)

        self.matrix = np.bmat(matrix).A

    def propose_out(self, projsin):
        """Propose input and output projection names given proposed input projection names ``projsin``."""
        projsin, projsout = super(PowerWindowMatrix,self).propose_out(projsin)
        if self.sum_wa:
            projsin = ProjectionNameCollection([proj for proj in projsout if proj.wa_order is not None])
            projsout = ProjectionNameCollection([proj.copy(wa_order=None) for proj in projsout])
        return projsin, projsout


def wigner3j_square(ellout, ellin, prefactor=True):
    r"""
    Return the coefficients corresponding to the product of two Legendre polynomials.

    Parameters
    ----------
    ellout : int
        Output order.

    ellin : int
        Input order.

    prefactor : bool, default=True
        Whether to include prefactor :math:`(2 \ell + 1)/(2 q + 1)` for window convolution.

    Returns
    -------
    ells : list
        List of mulipole orders.

    coeffs : list
        List of corresponding window coefficients.
    """
    qvals, coeffs = [], []

    def G(p):
        """
        Return the function G(p), as defined in Wilson et al 2015.
        See also: WA Al-Salam 1953
        Taken from https://github.com/nickhand/pyRSD.

        Parameters
        ----------
        p : int
            Order.

        Returns
        -------
        numer, denom: int
            The numerator and denominator.
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
        coeffs.append(numer*1./denom)
        qvals.append(q)

    return qvals[::-1], coeffs[::-1]
