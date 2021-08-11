"""Implementation of Finger-of-God terms."""

import numpy as np

from cosmopipe.lib import utils
from cosmopipe.lib.utils import BaseClass


class RegisteredFoG(type):

    """Metaclass registering FoG classes."""
    _registry = {}

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)
        meta._registry[name[:-len('FoG')].lower()] = cls
        return cls


class GaussianFoG(metaclass=RegisteredFoG):

    """Gaussian Finger-of-God"""

    def __call__(self, k, mu, sigmav=0., grid=True):
        """
        Return Gaussian FoG damping.

        Parameters
        ----------
        k : array
            Wavenumbers.

        mu : array
            Angle to the line-of-sight.

        sigmav : float, default=0
            Velocity dispersion.

        grid : bool
            Whether input ``k``, ``mu`` should be interpreted as a grid,
            in which case the output will be an array of shape ``(k.size, mu.size)``.

        Returns
        -------
        fog : array
        """
        k,mu = utils.enforce_shape(k,mu,grid=grid)
        return np.exp(-k**2*mu**2*sigmav**2)


class LorentzianFoG(metaclass=RegisteredFoG):

    """Lorentzian Finger-of-God."""

    def __call__(self, k, mu, sigmav=0., grid=True):
        """
        Return Lorentzian FoG damping.

        Parameters
        ----------
        k : array
            Wavenumbers.

        mu : array
            Angle to the line-of-sight.

        sigmav : float, default=0
            Velocity dispersion.

        grid : bool
            Whether input ``k``, ``mu`` should be interpreted as a grid,
            in which case the output will be an array of shape ``(k.size, mu.size)``.

        Returns
        -------
        fog : array
        """
        k,mu = utils.enforce_shape(k,mu,grid=grid)
        return 1./(1 + k**2*mu**2*sigmav**2)**2


def get_FoG(name):
    return RegisteredFoG._registry[name.lower()]()
