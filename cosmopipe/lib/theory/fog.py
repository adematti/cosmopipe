import numpy as np

from cosmopipe.lib.utils import BaseClass


class BaseFoG(BaseClass):

    pass


class GaussianFoG(BaseFoG):

    def __call__(self, k, mu, sigmav=0.):
        return np.exp(-k[:,None]**2*mu**2*sigmav**2)


class LorentzianFoG(BaseFoG):

    def __call__(self, k, mu, sigmav=0.):
        return 1./(1 + k[:,None]**2*mu**2*sigmav**2)**2


BaseFoG.registry = {}
for cls in BaseFoG.__subclasses__():
    name = cls.__name__[:-len('FoG')].lower()
    cls.name = name
    BaseFoG.registry[name] = cls


def get_FoG(name):
    return BaseFoG.registry[name]()
