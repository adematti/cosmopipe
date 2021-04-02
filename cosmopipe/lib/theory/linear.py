import numpy as np

from .base import BasePTModel


class LinearModel(BasePTModel):

    def pk_mu(self, k, mu=0., b1=1., shotnoise=0., growth_rate=None, **kwargs):
        beta = (growth_rate if growth_rate is not None else self.cosmo['growth_rate'])/b1
        pk_lin = self.pk_linear(k=k)
        toret = self.FoG(k=k,mu=mu,**kwargs)*(1. + beta*mu**2)**2 * b1**2 * pk_lin[:,None] + shotnoise
        if np.isscalar(mu): return toret[:,0]
        return toret
