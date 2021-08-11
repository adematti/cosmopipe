"""Classes to perform model integrations (typically over cosine angle :math:`\mu`)."""

import numpy as np
from scipy import special

from cosmopipe.lib import utils
from cosmopipe.lib.utils import BaseClass


def weights_trapz(x):
    """Return weights for trapezoidal integration."""
    return np.concatenate([[x[1]-x[0]],x[2:]-x[:-2],[x[-1]-x[-2]]])/2.


class BaseMultipoleIntegration(BaseClass):

    """Base class to perform integration over Legendre polynomials."""

    def __init__(self, mu=100, ells=(0,2,4), sym=False):
        r"""
        Initialize :class:`BaseMultipoleIntegration`.

        Parameters
        ----------
        mu : int, array, default=100
            :math:`\mu` cosine angle to integrate on.
            :math:`\mu`-coordinate array or number of :math:`\mu`-bins.

        ells : tuple
            Multipole orders.

        sym : bool, default=False
            Whether integration is symmetric around :math:`\mu = 0`.
            In this case, and if input ``mu`` is the number of bins, only integrates between 0 and 1.
        """
        self.mu = mu
        if np.ndim(self.mu) == 0:
            self.mu = np.linspace(0. if sym else -1.,1.,self.mu+1)
        self.ells = ells
        self.set_mu_weights()

    def __call__(self, array):
        """Integrate input array."""
        return np.sum(array*self.weights[:,None,:],axis=-1)


# TODO: implement gauss-legendre integration
class TrapzMultipoleIntegration(BaseMultipoleIntegration):

    """Class performing trapzeoidal integration over Legendre polynomials."""

    def set_mu_weights(self):
        """Set weights for trapezoidal integration."""
        muw_trapz = weights_trapz(self.mu)
        from scipy import special
        self.weights = np.array([muw_trapz*(2*ell+1.)*special.legendre(ell)(self.mu) for ell in self.ells])/(self.mu[-1]-self.mu[0])


class BaseMuWedgeIntegration(BaseClass):

    r"""Base class to perform integration over :math:`\mu`-wedges."""

    def __init__(self, mu=100, muwedges=3, sym=False):
        r"""
        Initialize :class:`BaseMuWedgeIntegration`.

        Parameters
        ----------
        mu : int, list
            :math:`\mu` cosine angle to integrate on.
            Total number of :math:`\mu`-bins, or list of :math:`\mu`-coordinate arrays (one for each :math:`\mu`-wedge).

        muwedges : int, list
            Number of wedges, or list of tuples of wedges lower and upper bounds.
            Used if ``mu`` is total number of bins.

        sym : bool, default=False
            Whether integration is symmetric around :math:`\mu = 0`.
            In this case, and if input ``mu`` is the number of bins, only use wedges between 0 and 1.
        """
        self.mu = mu
        if np.ndim(self.mu) == 0:
            if np.ndim(muwedges) == 0:
                muwedges = np.linspace(0. if sym else -1.,1.,muwedges+1)
                muwedges = zip(muwedges[:-1],muwedges[1:])
            self.mu = [np.linspace(*muwedge,mu//len(muwedges)) for muwedge in muwedges]
        self.set_mu_weights()

    def __call__(self, array):
        """Integrate input array."""
        return np.sum(array*self.weights[:,None,:],axis=-1)


# TODO: implement gauss-legendre integration
class TrapzMuWedgeIntegration(BaseMuWedgeIntegration):

    r"""Class performing trapzeoidal integration over :math:`\mu`-wedges."""

    def set_mu_weights(self):
        """Set weights for trapezoidal integration."""
        self.weights = np.array([weights_trapz(mu)/(mu[-1]-mu[0]) for mu in self.mu])


class MultipoleExpansion(BaseClass):

    """Class performing Legendre expansion of multipoles."""

    def __init__(self, input_model=None):
        """
        Initialize :class:`MultipoleExpansion`.

        Parameters
        ----------
        model : BaseModel, callable
            Input model.

        basis : ProjectionBasis, default=None
            Projection basis. If ``None`` uses :attr:`BaseModel.basis` attribute of input model.
        """
        self.input_model = model
        self.basis = self.input_basis = basis if basis is not None else self.input_model.basis
        self.legendre = [special.legendre(ell) for ell in self.basis.projs]
        self.basis = self.input_basis.copy()
        self.basis.mode = ProjectionBasis.MUWEDGE

    def __call__(self, x, mu, grid=True, **kwargs):
        """
        Parameters
        ----------
        x : array
            x-coordinates (:math:`k` or :math:`s`).

        mu : array
            Angle to the line-of-sight.

        grid : bool
            Whether input ``x``, ``mu`` should be interpreted as a grid,
            in which case the output will be arrays of shape ``(x.size, mu.size)``.

        kwargs : dict
            Arguments for input model.

        Returns
        -------
        model : array
        """
        y = self.input_model(x)
        x,mu = utils.enforce_shape(x,mu,grid=grid)
        toret = 0
        for y_,leg in zip(y,self.legendre): toret += y_*leg(mu)
        return toret


class MultipoleToMultipole(BaseClass):

    r"""Class mapping multipoles to multipoles, i.e. simply implementing :math:`\delta_{\ell\ell^{\prime}}`."""

    def __init__(self, ellsin=(0,2,4), ellsout=(0,2,4)):
        """
        Initialize :class:`MultipoleToMultipole`.

        Parameters
        ----------
        ellsin : tuple, default=(0,2,4)
            Input multipole orders.

        ellsout : tuple, default=(0,2,4)
            Output multipole orders.
        """
        self.weights = np.zeros((len(ellsin),len(ellsout)),dtype='f8')
        ellsout = np.array(ellsout)
        for illin,ellin in enumerate(ellsin):
            self.weights[illin,ellsout == ellin] = 1.

    def __call__(self, array):
        """Map array to :attr:`ellsout` multipoles."""
        return array.dot(self.weights)


class MultipoleToMuWedge(BaseClass):

    r"""Class mapping multipoles to :math:`\mu`-wedges."""

    def __init__(self, ellsin=(0,2,4), muwedges=3, sym=False):
        r"""
        Initialize :class:`MultipoleToMuWedge`.

        Parameters
        ----------
        ellsin : tuple, default=(0,2,4)
            Input multipole orders.

        muwedges : int, list
            Number of wedges, or list of tuples of wedges lower and upper bounds.

        sym : bool, default=False
            Whether integration is symmetric around :math:`\mu = 0`.
            In this case, and if input ``muwedges`` is the number of wedges, only use wedges between 0 and 1.
        """
        if np.ndim(muwedges) == 0:
            muwedges = np.linspace(0. if sym else -1.,1.,muwedges+1)
            muwedges = zip(muwedges[:-1],muwedges[1:])
        if np.ndim(muwedges[0]) == 0:
            muwedges = [muwedges]
        integlegendre = [special.legendre(ell).integ() for ell in ellsin]
        mulow,muup = np.array([muwedge[0] for muwedge in muwedges]),np.array([muwedge[-1] for muwedge in muwedges])
        muwidth = muup - mulow
        self.weights = np.array([(poly(muup) - poly(mulow))/muwidth for poly in integlegendre])

    def __call__(self, array):
        """Map array to :math:`\mu`-wedges."""
        return array.dot(self.weights)



def MultipoleIntegration(integration=None):
    """Convenient function to instantiate classes for multipole integration, with arguments ``integration``."""
    default = {'mu':100,'ells':(0,2,4)}
    if integration is None:
        integration = {}
    if isinstance(integration,dict):
        return TrapzMultipoleIntegration(**{**default,**integration})
    return integration


def MuWedgeIntegration(integration=None):
    r"""Convenient function to instantiate classes for :math:`\mu`-wedge integration, with arguments ``integration``."""
    default = {'mu':100,'muwedges':3}
    if integration is None:
        integration = {}
    if isinstance(integration,dict):
        return TrapzMuWedgeIntegration(**{**default,**integration})
    return integration
