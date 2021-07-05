import logging

import numpy as np
from scipy import special
import Corrfunc

from cosmopipe.lib import utils
from cosmopipe.lib.data_vector import BinnedStatistic, BinnedProjection, ProjectionName


class PairCount(BinnedStatistic):

    def __init__(self, wnpairs, total_wnpairs=1.):
        super(PairCount,self).__init__({'wnpairs':wnpairs},attrs={'total_wnpairs':total_wnpairs})

    def set_total_wnpairs(self, w1, w2=None):
        if w2 is not None:
            self.attrs['total_wnpairs'] = np.sum(w1)*np.sum(w2)
        else:
            self.attrs['total_wnpairs'] = np.sum(w1)**2 - np.sum(w1**2)

    def normalized(self):
        return self.data['wnpairs']/self.attrs['total_wnpairs']


class CorrelationFunctionEstimator(BinnedProjection):

    def __init__(self, D1D2, R1R2, D1R2=None, D2R1=None, data=None, edges=None, attrs=None, **kwargs):
        data = (data or {}).copy()
        attrs = attrs or {}
        if D2R1 is None:
            D2R1 = D1R2
        for key,value in zip(['D1D2','R1R2','D1R2','D2R1'],[D1D2,R1R2,D1R2,D2R1]):
            if isinstance(value,np.ndarray):
                data[key] = value
                attrs.setdefault('{}_total_wnpairs'.format(key),1.)
            elif value is not None:
                data[key] = value['wnpairs']
                attrs['{}_total_wnpairs'.format(key)] = value.attrs['total_wnpairs']
        data.setdefault('D2R1',data.get('D1R2',None))
        data = {name:value for name,value in data.items() if value is not None}
        attrs['columns_to_sum'] = [name for name in ['D1D2','R1R2','D1R2','D2R1'] if data.get(name,None) is not None]
        attrs['weights'] = 'R1R2'
        attrs['x'] = list(edges.keys())
        attrs['y'] = 'corr'
        super(CorrelationFunctionEstimator,self).__init__(data,attrs=attrs,edges=edges,**kwargs)
        self.set_corr()

    def total_wnpairs(self, key):
        return self.attrs['{}_total_wnpairs'.format(key)]

    def normalized(self, key):
        return self[key]/self.attrs['{}_total_wnpairs'.format(key)]

    def project_to_wp(self):
        new = self.copy()
        dpi = np.diff(self.edges['pi'])
        new.data = {}
        x = new.attrs['x'] = self.attrs['x'][0]
        new.edges = {x:self.edges[x]}
        new.set_x(self[x].mean(axis=-1))
        new.set_y(2*(self['corr']*dpi).sum(axis=-1))
        return new

    def project_to_muwedges(self, muwedges):
        toret = []
        if np.ndim(muwedges) == 0:
            muwedges = [(imu*1./muwedges,(imu+1)*1./muwedges) for imu in range(muwedges)]
        isscalar = np.ndim(muwedges[0]) == 0
        if isscalar: muwedges = [muwedges]
        for muwedge in muwedges:
            new = self.copy()
            new.proj = self.proj.copy()
            new.proj.mode = ProjectionName.MUWEDGE
            new.proj.proj = tuple(muwedge)
            new.set_new_edges(muwedge,dims='mu')
            new.squeeze(dims='mu')
            toret.append(new)
        if isscalar:
            toret = toret[0]
        return toret

    def project_to_multipoles(self, ells=(0,2,4)):
        from scipy import special
        toret = []
        isscalar = np.ndim(ells) == 0
        if isscalar: ells = [ells]
        for ell in ells:
            new = self.copy()
            new.data = {}
            x = self.attrs['x'][0]
            new.attrs['x'] = (x,)
            new.edges = {x:self.edges[x]}
            new.proj = self.proj.copy()
            new.proj.mode = ProjectionName.MULTIPOLE
            new.proj.proj = ell
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

    def set_corr(self):
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

    def set_corr(self):
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
