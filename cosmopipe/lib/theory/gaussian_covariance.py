import logging

import numpy as np
from scipy import special

from cosmopipe.lib.data import CovarianceMatrix
from .projection import ProjectionName, DataVectorProjection


class GaussianPkCovarianceMatrix(CovarianceMatrix):

    logger = logging.getLogger('GaussianPkCovarianceMatrix')

    def __init__(self, kedges, k=None, projs=('ell_0','ell_2','ell_4'), volume=(1e3)**3, shotnoise=0, integration=None):

        if k is None:
            k = 3./4.*(kedges[1:]**4-kedges[:-1]**4)/(kedges[1:]**3-kedges[:-1]**3)
        volume_k = 4.*np.pi/3.*(kedges[1:]**3-kedges[:-1]**3)
        nk = volume_k*volume/(8.*np.pi**3)
        self.projection = DataVectorProjection(k,projs,basemodel='xmu',integration=integration)
        self.attrs = {'shotnoise':shotnoise,'nk':nk}

    def run(self, pk_mu, **kwargs):
        mean = self.projection(pk_mu,concatenate=False)
        covariance = []

        for iproj1,proj1 in enumerate(self.projection.projnames):

            ell1 = 0
            dmu = 1.
            if proj1.type == 'multipole': ell1 = proj1.proj
            elif proj1.type == 'muwedge': dmu = abs(proj.proj[1] - proj.proj[0])
            def integrand(x,mu):
                return 2.*(2.*ell1+1.)/dmu/self.attrs['nk'][:,None] * (pk_mu(x,mu) + self.attrs['shotnoise'])**2 * special.legendre(ell1)(mu)

            line = self.projection(integrand,concatenate=False)
            covariance.append(np.concatenate([np.diag(li) for li in line],axis=-1))

        covariance = np.concatenate(covariance,axis=0)
        super(GaussianPkCovarianceMatrix,self).__init__(covariance,x=self.projection.x[0],mean=mean,mapping_proj=self.projection.projnames,**self.attrs)
