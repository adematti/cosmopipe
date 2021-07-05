import numpy as np

from cosmopipe.lib import utils
from cosmopipe.lib.utils import BaseClass


class BaseFoG(BaseClass):

    pass


class GaussianFoG(BaseFoG):

    def __call__(self, k, mu, sigmav=0., grid=True):
        k,mu = utils.enforce_shape(k,mu,grid=grid)
        return np.exp(-k**2*mu**2*sigmav**2)


class LorentzianFoG(BaseFoG):

    def __call__(self, k, mu, sigmav=0., grid=True):
        k,mu = utils.enforce_shape(k,mu,grid=grid)
        return 1./(1 + k**2*mu**2*sigmav**2)**2


BaseFoG.registry = {}
for cls in BaseFoG.__subclasses__():
    name = cls.__name__[:-len('FoG')].lower()
    cls.name = name
    BaseFoG.registry[name] = cls


def get_FoG(name):
    return BaseFoG.registry[name]()
