"""Implementation of Alcock-Paczynski transforms."""

import numpy as np

from cosmopipe.lib import utils
from cosmopipe.lib.utils import BaseClass

from .base import ProjectionBasis, BaseModel
from .integration import MultipoleExpansion


class AnisotropicScaling(BaseModel):

    """Class applying anisotropic scaling of the theory model (correlation function or power spectrum)."""

    def __init__(self, model=None, basis=None):
        r"""
        Initialize :class:`AnisotropicScaling`.

        Parameters
        ----------
        model : BaseModel, callable
            Input model.

        basis : ProjectionBasis, default=None
            Projection basis. If ``None`` uses :attr:`BaseModel.basis` attribute of input model.
            If ``basis.mode`` is ``'multipole'``, model is first expanded onto Legendre polynomials to get :math:'\mu' dependence.
        """
        self.input_model = model
        self.basis = self.input_basis = basis if basis is not None else self.input_model.basis
        if self.input_basis.mode == ProjectionBasis.MULTIPOLE:
            self.multipole_expansion = MultipoleExpansion(self.input_model)
            self.basis = self.multipole_expansion.basis
        self.set_scaling()

    def set_scaling(self, qpar=1, qperp=1):
        """Set scaling parameters, along ``qpar`` and perpendicular ``qperp`` to the line-of-sight."""
        self.qpar = qpar
        self.qperp = qperp
        self.qap = qpar/qperp
        self.qiso = (self.qperp**2*self.qpar)**(1./3.)

    def kmu_scaling(self, k, mu, grid=True):
        r"""
        Apply anisotropic scaling to :math:`k, \mu` coordinates.

        Parameters
        ----------
        k : array
            Wavenumbers.

        mu : array
            Angle to the line-of-sight.

        grid : bool
            Whether input ``k``, ``mu`` should be interpreted as a grid,
            in which case the output will be arrays of shape ``(k.size, mu.size)``.

        Returns
        -------
        kap : array
            Wavenumbers after rescaling.

        muap ; array
            Angle to the line-of-sight after rescaling.
        """
        factor_ap = np.sqrt(1 + mu**2*(1./self.qap**2-1))
        k,mu = utils.enforce_shape(k,mu,grid=grid)
        # Beutler 2016 (arXiv: 1607.03150) eq 44
        kap = k/self.qperp*factor_ap
        # Beutler 2016 (arXiv: 1607.03150) eq 45
        muap = mu/self.qap/factor_ap
        return kap, muap

    def smu_scaling(self, s, mu, grid=True):
        r"""
        Apply anistropic scaling to :math:`s, \mu` coordinates.

        Parameters
        ----------
        s : array
            Separations.

        mu : array
            Angle to the line-of-sight.

        grid : bool
            Whether input ``s``, ``mu`` should be interpreted as a grid,
            in which case the output will be arrays of shape ``(s.size, mu.size)``.

        Returns
        -------
        sap : array
            Wavenumbers after rescaling.

        muap ; array
            Angle to the line-of-sight after rescaling.
        """
        factor_ap = np.sqrt(mu**2*(self.qap**2-1) + 1)
        # Hou 2018 (arXiv: 2007.08998) eq 8
        sap = s*self.qperp*factor_ap
        muap = mu*self.qap/factor_ap
        return sap, muap

    def eval(self, x, mu=0., grid=True, **kwargs):
        """
        Evaluate model in rescaled coordinates.

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
        if self.input_basis.mode == ProjectionBasis.MULTIPOLE:
            self.multipole_expansion.input_model = self.input_model
            input_model = self.multipole_expansion
        else:
            input_model = self.input_model
        if self.basis.space == ProjectionBasis.CORRELATION:
            sap, muap = self.smu_scaling(x,mu,grid=grid)
            return input_model(sap,mu=muap,grid=False,**kwargs)
        kap, muap = self.kmu_scaling(x,mu,grid=grid)
        return 1./self.qiso**3*input_model(kap,mu=muap,grid=False,**kwargs) # in Fourier space, extra jacobian



class IsotropicScaling(BaseModel):

    """Class applying isotropic scaling of the theory model (correlation function or power spectrum)."""

    def __init__(self, model=None, basis=None, pivot=1./3.):
        """
        Initialize :class:`AnisotropicScaling`.

        Parameters
        ----------
        model : BaseModel, callable
            Input model.

        basis : ProjectionBasis, default=None
            Projection basis. If ``None`` uses :attr:`BaseModel.basis` attribute of input model.

        pivot : float, default=1./3.
            Pivot square cosine angle that defines isotropic scaling compared to the anistropic (AP) effect.
        """
        self.input_model = model
        self.basis = basis if basis is not None else self.input_model.basis
        self.pivot = pivot
        self.set_scaling()

    def set_scaling(self, qpar=1., qperp=1.):
        """Set scaling parameters, along ``qpar`` and perpendicular ``qperp`` to the line-of-sight."""
        self.qiso = qpar**self.pivot*qperp**(1.-self.pivot)
        self.qap = qpar/qperp

    def anisotropic_scaling(self):
        """Return new ``qpar`` and perpendicular ``qperp`` scaling parameters to apply after isotropic scaling."""
        return self.qap**(1.-self.pivot), self.qap**(-self.pivot)

    def k_scaling(self, k):
        """Apply isotropic scaling to input wavenumbers."""
        return k/self.qiso

    def s_scaling(self, s):
        """Apply isotropic scaling to input separations."""
        return s/self.qiso

    def eval(self, x, **kwargs):
        """
        Evaluate model in rescaled coordinates.

        Parameters
        ----------
        x : array
            x-coordinates (:math:`k` or :math:`s`).

        kwargs : dict
            Arguments for input model.

        Returns
        -------
        model : array
        """
        if self.basis.space == ProjectionBasis.CORRELATION:
            return self.input_model(self.s_scaling(x),**kwargs)
        return 1/self.qiso**3*self.input_model(self.k_scaling(x),**kwargs) # in Fourier space, extra jacobian
