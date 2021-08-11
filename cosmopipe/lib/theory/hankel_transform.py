"""Implementation of Hankel transform."""

import numpy as np
from scipy import interpolate

from cosmoprimo import PowerToCorrelation, CorrelationToPower

from cosmopipe.lib.data_vector import DataVector, ProjectionName
from .base import BaseModel, ProjectionBasis
from .evaluation import ModelEvaluation


class HankelTransform(BaseModel):

    """Class performing (forward and backward) Hankel transforms."""

    def __init__(self, model=None, basis=None, nx=None, ells=None, q=0, integration=None):
        """
        Initialize :class:`HankelTransform`.

        Parameters
        ----------
        model : BaseModel, callable
            Input model.

        basis : ProjectionBasis, default=None
            Projection basis. If ``None`` uses :attr:`BaseModel.basis` attribute of input model.
            If ``basis.mode`` is ``'muwedge'``, model is first integrated onto Legendre polynomials to get multipoles.

        nx : int, default=None
            Number of log-space points. If ``None``, defaults to length of ``basis.x``.

        ells : tuple, default=None
            Multipole orders. If ``None``, defaults to ``basis.projs`` if ``basis.mode`` is ``'muwedge'``, else ``(0,2,4)``.

        q : int, default=0
            Power-law tilt to regularize Hankel transforms.

        integration : dict
            If ``basis.mode`` is ``'muwedge'``, options for integration over Legendre polynomials, see :class:`ModelEvaluation`.
        """
        self.input_model = model
        self.input_basis = basis if basis is not None else self.input_model.basis
        x = self.input_basis.x
        xmin,xmax = x.min(),x.max()
        self.x = np.logspace(np.log10(xmin),np.log10(xmax),nx or len(x))
        self.set_damping()
        if self.input_basis.mode == ProjectionBasis.MUWEDGE:
            self.ells = ells or (0,2,4)
        else:
            self.ells = ells or self.input_basis.projs
        projs = [ProjectionName((ProjectionName.MULTIPOLE,ell)) for ell in self.ells]
        self.evaluation = ModelEvaluation(self.x,projs=projs,model_bases=self.input_basis,integration=integration)
        self.basis = self.input_basis.copy()
        if self.input_basis.space == ProjectionBasis.POWER:
            self.fftlog = PowerToCorrelation(self.x,ell=self.ells,q=q,lowring=False,xy=1)
            self.basis.space = ProjectionBasis.CORRELATION
        if self.input_basis.space == ProjectionBasis.CORRELATION:
            self.fftlog = CorrelationToPower(self.x,ell=self.ells,q=q,lowring=False,xy=1)
            self.basis.space = ProjectionBasis.POWER
        self.basis.x = self.fftlog.y[0]
        self.basis.mode = ProjectionBasis.MULTIPOLE
        self.basis.projs = self.ells

    def set_damping(self):
        """Set artificial damping of the input model to regularize integration."""
        x = self.x
        self.damping = 1.
        if self.input_basis.space == ProjectionBasis.POWER:
            self.damping = np.ones(x.size,dtype='f8')
            cutoff = 2.
            high = x>cutoff
            self.damping[high] *= np.exp(-(x[high]/cutoff-1.)**2)
            cutoff = 1e-4
            low = x<cutoff
            self.damping[low] *= np.exp(-(cutoff/x[low]-1.)**2)

    def eval(self, x, **kwargs):
        """
        Evaluate Hankel-transformed model.

        Parameters
        ----------
        x : array
            x-coordinates (wavenumbers or separations).

        kwargs : dict
            Arguments for :meth:`ModelEvaluation.__call__`.

        Returns
        -------
        model : array
        """
        modelell = self.evaluation(self.input_model,concatenate=False,**kwargs)*self.damping
        modelell = self.fftlog(modelell)[-1].T
        return interpolate.interp1d(self.basis.x,modelell,axis=0,kind='cubic',bounds_error=True,assume_sorted=True)(x)
