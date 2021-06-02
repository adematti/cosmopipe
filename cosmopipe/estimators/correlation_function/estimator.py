import logging

import numpy as np
from scipy import special
import Corrfunc

from cosmopipe.lib import utils
from cosmopipe.lib.utils import BaseClass


class PairCount(BaseClass):

    def __init__(self, wnpairs, total_wnpairs=1.):
        self.wnpairs = wnpairs
        self.total_wnpairs = total_wnpairs

    def set_total_wnpairs(self, w1, w2=None):
        if w2 is not None:
            self.total_wnpairs = np.sum(w1)*np.sum(w2)
        else:
            self.total_wnpairs = np.sum(w1)**2 - np.sum(w1**2)

    def __getstate__(self):
        state = {'wnpairs':self.wnpairs,'total_wnpairs':self.total_wnpairs}

    def __setstate__(self, state):
        self.__dict__.update(state)

    def normalized(self):
        return self.wnpairs/self.total_wnpairs

    @property
    def shape(self):
        return self.wnpairs.shape


class BaseEstimator(BaseClass):

    def __init__(self, D1D2, R1R2, D1R2=None, D2R1=None, edges=None, sep=None, **attrs):
        self.D1D2 = D1D2
        self.R1R2 = R1R2
        self.D1R2 = D1R2
        if D2R1 is None:
            self.D2R1 = self.D1R2
        if not isinstance(edges,(tuple,list)):
            self.edges = (edges,)
        else:
            self.edges = tuple(edges)
        if not isinstance(sep,(tuple,list)):
            self.sep = (sep,)
        else:
            self.sep = tuple(sep)
        self.attrs = attrs
        self.set_corr()

    def __getstate__(self):
        state = {}
        for key in ['D1D2','D1R2','D2R1','R1R2']:
            if hasattr(self,key): state[key] = getattr(self,key).__getstate__()
        for key in ['edges','sep','corr','attrs']:
            if hasattr(self,key): state[key] = getattr(self,key)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        for key in ['D1D2','D1R2','D2R1','R1R2']:
            if hasattr(self,key): state[key] = PairCount.from_state(getattr(self,key))

    def rebin(self, new_edges):
        self.edges = new_edges
        new_edges = tuple(np.interp(ne,e,np.arange(len(e))) for e,ne in zip(self.edges,new_edges)) # fraction of index numbers
        weights = utils.rebin_ndarray(self.R1R2.wnpairs,new_edges,interpolation='linear')
        self.sep = tuple(utils.rebin_ndarray(sep*self.R1R2.wnpairs,new_edges,interpolation='linear')/weights for sep in self.sep)
        self.corr = utils.rebin_ndarray(self.corr*self.R1R2.wnpairs,new_edges,interpolation='linear')/weights


class NaturalEstimator(BaseEstimator):

    def set_corr():
        nonzero = self.R1R2.wnpairs > 0
        # init
        self.corr = np.empty(self.D1D2.shape)
        self.corr[...] = np.nan

        # the Landy - Szalay estimator
        # (DD - DR - RD + RR) / RR
        DD = self.D1D2.normalized()[nonzero]
        RR = self.R1R2.normalized()[nonzero]
        corr = DD/RR - 1
        self.corr[nonzero] = corr[:]


class LandySzalayEstimator(BaseEstimator):

    def set_corr():
        nonzero = self.R1R2.wnpairs > 0
        # init
        self.corr = np.empty(self.D1D2.shape)
        self.corr[...] = np.nan

        # the Landy - Szalay estimator
        # (DD - DR - RD + RR) / RR
        DD = self.D1D2.normalized()[nonzero]
        DR = self.D1R2.normalized()[nonzero]
        RD = self.D2R1.normalized()[nonzero]
        RR = self.R1R2.normalized()[nonzero]
        corr = (DD - DR - RD)/RR + 1
        self.corr[nonzero] = corr[:]


def project_to_multipoles(estimator, ells=(0,2,4)):
    from scipy import special
    toret = []
    dmu = np.diff(estimator.edges[1])
    for ell in ells:
        legendre = (2*ell + 1) * special.legendre(ell)(self.sep[1])
        toret.append(np.sum(estimator.corr*legendre*dmu,axis=-1)/np.sum(dmu))
    return np.mean(self.sep[0],axis=-1), np.array(toret).T
