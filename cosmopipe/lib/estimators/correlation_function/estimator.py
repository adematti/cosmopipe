"""Implementations of natural and Landy-Szalay estimators."""

import logging

import numpy as np
from scipy import special
import Corrfunc

from cosmopipe.lib.utils import BaseClass
from cosmopipe.lib import utils
from cosmopipe.lib.data_vector import BinnedStatistic, BinnedProjection, ProjectionName


class PairCount(BaseClass):
    """
    Class holding pair counts.

    Attributes
    ----------
    wnpairs : array
        Weighted number of pairs (as a function of separation).

    total_wnpairs : float
        Total number of pairs.
    """
    def __init__(self, wnpairs, total_wnpairs=1.):
        """
        Initialize :class:`PairCount`.

        Parameters
        ----------
        wnpairs : array
            Weighted number of pairs (as a function of separation).

        total_wnpairs : float
            Total number of pairs.
        """
        self.wnpairs = wnpairs
        self.total_wnpairs = total_wnpairs

    def set_total_wnpairs(self, w1, w2=None):
        """
        Set total weighted number of pairs.

        Parameters
        ----------
        w1 : array
            Weights to first catalog.

        w2 : array, default=None
            Weights for second catalog, if cross-correlation.
            If ``None``, set number of pairs for an auto-correlation.
        """
        if w2 is not None:
            self.total_wnpairs = np.sum(w1)*np.sum(w2)
        else:
            self.total_wnpairs = np.sum(w1)**2 - np.sum(w1**2)

    def normalized(self):
        """Normalized pair counts, i.e. (weighted) pair counts divided by total weighted pairs."""
        return self.wnpairs/self.total_wnpairs


class CorrelationFunctionEstimator(BinnedProjection):

    """Base class for correlation function estimation from pair counts."""

    def __init__(self, D1D2=None, R1R2=None, D1R2=None, D2R1=None, data=None, dims=None, edges=None, attrs=None, **kwargs):
        """
        Initialize :class:`CorrelationFunctionEstimator`.

        Parameters
        ----------
        D1D2 : array, PairCount, default=None
            DD pair counts. If not provided, must be in ``data``.

        R1R2 : array, PairCount, default=None
            RR pair counts. If not provided, must be in ``data``.

        D1R2 : array, PairCount, default=None
            DR pair counts. Not required for :class:`NaturalEstimator`.

        R1D2 : array, PairCount, default=None
            RD pair counts. Not required for :class:`NaturalEstimator` or for autocorrelation.

        data : dict, default=None
            Dictionary of pair counts, if not provided previously.

        dims : tuple
            Dimensions, e.g. ``('s','mu')``.

        edges : dict, list, default=None
            Dictionary of edges, or list of edges corresponding to ``dims``.
            If ``None``, no edges considered.

        attrs : dict
            Other attributes.

        kwargs : dict
            Other arguments for :meth:`BinnedProjection.__init__`.
        """
        data = (data or {}).copy()
        attrs = (attrs or {}).copy()
        for key,value in zip(['D1D2','R1R2','D1R2','D2R1'],[D1D2,R1R2,D1R2,D2R1]):
            if isinstance(value,np.ndarray):
                data[key] = value
                attrs.setdefault('{}_total_wnpairs'.format(key),1.)
            elif value is not None:
                data[key] = value.wnpairs
                attrs['{}_total_wnpairs'.format(key)] = value.total_wnpairs
        data.setdefault('D2R1',data.get('D1R2',None))
        attrs.setdefault('D2R1_total_wnpairs',attrs.get('D1R2_total_wnpairs',None))
        data = {name:value for name,value in data.items() if value is not None}
        attrs['columns_to_sum'] = [name for name in ['D1D2','R1R2','D1R2','D2R1'] if data.get(name,None) is not None]
        attrs['weights'] = 'R1R2'
        attrs['y'] = 'corr'
        super(CorrelationFunctionEstimator,self).__init__(data,attrs=attrs,dims=dims,edges=edges,**kwargs)
        self.set_corr()

    def total_wnpairs(self, name):
        """Return total weighted number of pairs ``name`` (e.g. ``'D1D2'``)."""
        return self.attrs['{}_total_wnpairs'.format(name)]

    def normalized(self, name):
        """Return normalized number of pairs ``name`` (e.g. ``'D1D2'``)."""
        return self[name]/self.attrs['{}_total_wnpairs'.format(name)]

    def project_to_wp(self):
        """
        Project estimated correlation to projected correlation function :math:`w_{p}(r_{p})`.

        Returns
        -------
        new : BinnedProjection
        """
        new = BinnedProjection.__new__(BinnedProjection)
        new.__dict__.update(self.copy().__dict__)
        new.proj = self.proj.copy()
        dpi = np.diff(self.edges['pi'])
        new.data = {}
        new.dims = self.dims[:1]
        x = new.dims[0]
        new.edges = {x:self.edges[x]}
        new.set_x(self[x].mean(axis=-1))
        new.set_y(2*(self['corr']*dpi).sum(axis=-1))
        return new

    def project_to_muwedges(self, muwedges):
        r"""
        Project estimated correlation on :math:`\mu`-wedges.

        Parameters
        ----------
        muwedges : int, list, tuple
            Number of wedges or (list of) wedges (tuple).

        Returns
        -------
        new : list, BinnedProjection
            If only one wedge required, return single :class:`BinnedProjection` instance.
            Else, list of :class:`BinnedProjection` corresponding to list of wedges.
        """
        toret = []
        if np.ndim(muwedges) == 0:
            muwedges = [(imu*1./muwedges,(imu+1)*1./muwedges) for imu in range(muwedges)]
        isscalar = np.ndim(muwedges[0]) == 0
        if isscalar: muwedges = [muwedges]
        for muwedge in muwedges:
            new = BinnedProjection.__new__(BinnedProjection)
            new.__dict__.update(self.copy().__dict__)
            new.proj = self.proj.copy(mode=ProjectionName.MUWEDGE,proj=tuple(muwedge))
            new.set_new_edges(muwedge,dims='mu')
            new.squeeze(dims='mu')
            toret.append(new)
        if isscalar:
            toret = toret[0]
        return toret

    def project_to_multipoles(self, ells=(0,2,4)):
        """
        Project estimated correlation on Legendre polynomials.

        Parameters
        ----------
        ells : int, list
            Order(s) of Legendre polynomials.

        Returns
        -------
        new : list, BinnedProjection
            If only one multipole required, return single :class:`BinnedProjection` instance.
            Else, list of :class:`BinnedProjection` corresponding to list of multipoles.
        """
        from scipy import special
        toret = []
        isscalar = np.ndim(ells) == 0
        if isscalar: ells = [ells]
        for ell in ells:
            new = BinnedProjection.__new__(BinnedProjection)
            new.__dict__.update(self.copy().__dict__)
            new.data = {}
            new.dims = self.dims[:1]
            x = new.dims[0]
            new.edges = {x:self.edges[x]}
            new.proj = self.proj.copy(mode=ProjectionName.MULTIPOLE,proj=ell)
            edges = self.edges['mu']
            dmu = np.diff(edges,axis=-1)
            poly = special.legendre(ell)(edges)
            legendre = (2*ell + 1) * (poly[1:] + poly[:-1])/2. * dmu
            new.set_x(np.mean(self[x],axis=-1))
            new.set_y(np.sum(self['corr']*legendre,axis=-1)/np.sum(dmu))
            toret.append(new)
        if isscalar:
            toret = toret[0]
        return toret


class NaturalEstimator(CorrelationFunctionEstimator):

    """Natural estimator of the correlation function: :math:`DD/RR - 1`."""

    def set_corr(self):
        """Set correlation estimation."""
        nonzero = self['R1R2'] != 0
        # init
        corr = np.empty(self.shape)
        corr[...] = np.nan

        # the Landy - Szalay estimator
        # (DD - DR - RD + RR) / RR
        DD = self.normalized('D1D2')[nonzero]
        RR = self.normalized('R1R2')[nonzero]
        tmp = DD/RR - 1
        corr[nonzero] = tmp[:]
        self['corr'] = corr


class LandySzalayEstimator(CorrelationFunctionEstimator):

    """Landy-Szalay estimator of the correlation function: :math:`(DD - 2DR + RR)/RR`."""

    def set_corr(self):
        """Set correlation estimation."""
        nonzero = self['R1R2'] != 0
        # init
        corr = np.empty(self.shape)
        corr[...] = np.nan

        # the Landy - Szalay estimator
        # (DD - DR - RD + RR) / RR
        DD = self.normalized('D1D2')[nonzero]
        DR = self.normalized('D1R2')[nonzero]
        RD = self.normalized('D2R1')[nonzero]
        RR = self.normalized('R1R2')[nonzero]
        tmp = (DD - DR - RD)/RR + 1
        corr[nonzero] = tmp[:]
        self['corr'] = corr
