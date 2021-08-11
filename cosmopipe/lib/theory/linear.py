"""Implementation of Kaiser RSD model."""

import numpy as np

from cosmopipe.lib import utils
from .base import BasePTModel


class LinearModel(BasePTModel):

    """Linear RSD Kaiser model."""

    def pk_mu(self, k, mu=0., b1=2., shotnoise=0., f=1., grid=True, **kwargs):
        """
        Return power spectrum at input ``k``, ``mu``.

        Parameters
        ----------
        k : array
            Wavenumbers.

        mu : array
            Angle to the line-of-sight.

        b1 : float
            Linear bias.

        shotnoise : float
            Shot noise.

        f : float
            Growth rate.

        grid : bool
            Whether input ``k``, ``mu`` should be interpreted as a grid,
            in which case the output will be an array of shape ``(k.size, mu.size)``.

        kwargs : dict
            Arguments for :attr:`FoG`.

        Returns
        -------
        pk_mu : array
        """
        beta = f/b1
        pk_lin = self.pk_linear(k=k)
        pk_lin, mu = utils.enforce_shape(pk_lin,mu,grid=grid)
        toret = self.FoG(k=k,mu=mu,grid=grid,**kwargs)*(1. + beta*mu**2)**2 * b1**2 * pk_lin + shotnoise
        return toret

    eval = pk_mu
