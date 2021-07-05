import numpy as np

from cosmopipe.lib import utils
from .base import BasePTModel


class LinearModel(BasePTModel):

    def pk_mu(self, k, mu=0., b1=2., shotnoise=0., f=1., grid=True, **kwargs):
        beta = f/b1
        pk_lin = self.pk_linear(k=k)
        pk_lin, mu = utils.enforce_shape(pk_lin,mu,grid=grid)
        toret = self.FoG(k=k,mu=mu,grid=grid,**kwargs)*(1. + beta*mu**2)**2 * b1**2 * pk_lin + shotnoise
        return toret

    eval = pk_mu
