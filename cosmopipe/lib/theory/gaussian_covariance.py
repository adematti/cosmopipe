import logging

import numpy as np
from scipy import special

from cosmopipe.lib.data import CovarianceMatrix
from .projection import ProjectionName, DataVectorProjection


class GaussianPkCovarianceMatrix(CovarianceMatrix):

    logger = logging.getLogger('GaussianPkCovarianceMatrix')

    def __init__(self, edges, k=None, projs=('ell_0','ell_2','ell_4'), volume=(1e3)**3, shotnoise=0, integration=None):

        if k is None:
            k = 3./4.*(edges[1:]**4-edges[:-1]**4)/(edges[1:]**3-edges[:-1]**3)
        volume_k = 4.*np.pi/3.*(edges[1:]**3-edges[:-1]**3)
        nk = volume_k*volume/(8.*np.pi**3)
        self.projection = DataVectorProjection(k,projs,model_base='muwedge',integration=integration)
        self.attrs = {'shotnoise':shotnoise,'nk':nk}

    def compute(self, pk_mu, **kwargs):
        mean = self.projection(pk_mu,concatenate=False)
        covariance = []

        for iproj1,proj1 in enumerate(self.projection.projs):

            ell1 = 0
            muwedge = (0.,1.)
            if proj1.mode == 'multipole': ell1 = proj1.proj
            elif proj1.mode == 'muwedge': muwedge = proj1.proj
            dmu = muwedge[1] - muwedge[0]

            def integrand(x, mu):
                mask = (mu >= muwedge[0]) & (mu <= muwedge[1])
                toret = 2.*(2.*ell1+1.)/dmu/self.attrs['nk'][:,None] * (pk_mu(x,mu) + self.attrs['shotnoise'])**2 * special.legendre(ell1)(mu)
                toret[:,~mask] = 0.
                return toret

            line = self.projection(integrand,concatenate=False)
            covariance.append(np.concatenate([np.diag(li) for li in line],axis=-1))

        covariance = np.concatenate(covariance,axis=0)
        covariance = (covariance + covariance.T)/2.
        super(GaussianPkCovarianceMatrix,self).__init__(covariance,x=self.projection.x,mean=mean,mapping_proj=self.projection.projs,**self.attrs)



class GaussianXiCovarianceMatrix(CovarianceMatrix):

    logger = logging.getLogger('GaussianXiCovarianceMatrix')

    def __init__(self, edges, s=None, k=None, projs=('ell_0','ell_2','ell_4'), volume=(1e3)**3, shotnoise=0, integration=None):

        if s is None:
            s = 3./4.*(edges[1:]**4-edges[:-1]**4)/(edges[1:]**3-edges[:-1]**3)
        self.s = s
        self.projection = DataVectorProjection(self.s,projs,model_base='muwedge',integration=integration)

        bessel_kernels = {}
        self.k = k
        if k is None: self.k = np.linspace(0.,1.,100)
        for ell in [proj.proj for proj in self.projection.projs]:
            kernel = []
            for edge in zip(edges[:-1],edges[1:]):
                dv = (edge[1]**3 - edge[0]**3)/3.
                s = np.linspace(edge[0],edge[1],10)
                kernel.append(np.trapz(s[:,None]**2*special.spherical_jn(ell,s[:,None]*self.k[None,:],x=s,axis=0))
            bessel_kernels[ell] = np.array(self.k**2*kernel)

        self.attrs = {'volume':volume,'shotnoise':shotnoise}

    def compute(self, pk_mu, xi_mu=None, **kwargs):

        mean = self.projection(xi_mu,concatenate=False)
        covariance = []

        for proj1,proj2 in itertools.product(self.projection.projs,self.projection.projs):

            ell1,ell2 = proj1.proj,proj2.proj
            mu = np.linspace(-1.,1.,100)

            def integrand(x, mu):
                toret = (2.*ell1+1.) * (2.*ell2+1.) / self.attrs['volume'] * (pk_mu(x,mu) + self.attrs['shotnoise'])**2 * special.legendre(ell1)(mu) * special.legendre(ell2)(mu)

            sigmak = np.trapz(integrand(self.k,mu),x=mu,axis=-1)
            cov = np.trapz(sigmak*bessel_kernels[ell1][None,...]*bessel_kernels[ell2][:,None,...],x=self.k,axis=-1)

            covariance.append(cov)

        covariance = np.concatenate(covariance,axis=0)
        covariance = (covariance + covariance.T)/2.
        super(GaussianXiCovarianceMatrix,self).__init__(covariance,x=self.projection.x,mean=mean,mapping_proj=self.projection.projs,**self.attrs)



class GaussianCovarianceMatrix(CovarianceMatrix):

    logger = logging.getLogger('GaussianCovarianceMatrix')

    def __init__(self, edges, x=None, projs=('ell_0','ell_2','ell_4'), model_base='muwedge', volume=(1e3)**3, shotnoise=0, integration=None):

        projs = [ProjectionName(projname) for projname in projs]
        projs_corr = [proj for proj in projs if proj.space == 'corr']
        projs_power = [proj for proj in projs if proj.space != 'corr']

        if s is None:
            s = 3./4.*(edges[1:]**4-edges[:-1]**4)/(edges[1:]**3-edges[:-1]**3)
        self.s = s
        self.projection = DataVectorProjection(x,projs,model_base=model_base,integration=integration)

        bessel_kernels = {}
        self.k = k
        if k is None: self.k = np.linspace(0.,1.,100)
        for ell in [proj.proj for proj in self.projection.projs]:
            kernel = []
            for edge in zip(edges[:-1],edges[1:]):
                dv = (edge[1]**3 - edge[0]**3)/3.
                s = np.linspace(edge[0],edge[1],10)
                kernel.append(np.trapz(s[:,None]**2*special.spherical_jn(ell,s[:,None]*self.k[None,:],x=s,axis=0))
            bessel_kernels[ell] = np.array(self.k**2*kernel)

        self.attrs = {'volume':volume,'shotnoise':shotnoise}

    def compute(self, pk_mu, xi_mu=None, **kwargs):
        if xi_mu is None:
            projection = DataVectorProjection(self.k,projs,model_base='muwedge',integration=integration)
            pk_ell = None

        mean = self.projection(xi_mu,concatenate=False)
        covariance = []

        for proj1,proj2 in itertools.product(self.projection.projs,self.projection.projs):

            ell1,ell2 = proj1.proj,proj2.proj
            mu = np.linspace(-1.,1.,100)

            def integrand(x, mu):
                toret = (2.*ell1+1.) * (2.*ell2+1.) / self.attrs['volume'] * (pk_mu(x,mu) + self.attrs['shotnoise'])**2 * special.legendre(ell1)(mu) * special.legendre(ell2)(mu)

            sigmak = np.trapz(integrand(self.k,mu),x=mu,axis=-1)
            cov = np.trapz(sigmak*bessel_kernels[ell1][None,...]*bessel_kernels[ell2][:,None,...],x=self.k,axis=-1)

            covariance.append(cov)

        covariance = np.concatenate(covariance,axis=0)
        covariance = (covariance + covariance.T)/2.
        super(GaussianXiCovarianceMatrix,self).__init__(covariance,x=self.projection.x,mean=mean,mapping_proj=self.projection.projs,**self.attrs)
