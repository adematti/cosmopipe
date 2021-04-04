import numpy as np

from cosmopipe.lib import utils
from .base import BasePTModel


class LinearModel(BasePTModel):

    def pk_mu(self, k, mu=0., b1=1., shotnoise=0., f=None, **kwargs):
        beta = (f if f is not None else self.cosmo['f'])/b1
        pk_lin = self.pk_linear(k=k)
        pk_lin,mu = utils.enforce_shape(pk_lin,mu)
        toret = self.FoG(k=k,mu=mu,**kwargs)*(1. + beta*mu**2)**2 * b1**2 * pk_lin + shotnoise
        if np.isscalar(mu): return toret[:,0]
        return toret
