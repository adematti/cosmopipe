import numpy as np
from scipy import interpolate

from cosmoprimo import PowerToCorrelation, CorrelationToPower

from cosmopipe.lib.data_vector import DataVector, ProjectionName
from .base import BaseModel, ProjectionBase
from .projection import ModelProjection


class HankelTransform(BaseModel):

    def __init__(self, model=None, base=None, nx=None, ells=None, q=1.5, integration=None):
        self.input_model = model
        self.input_base = base if base is not None else self.input_model.base
        x = self.input_base.x
        xmin,xmax = x.min(),x.max()
        self.x = np.logspace(np.log10(xmin),np.log10(xmax),nx or len(x))
        self.set_damping()
        if self.input_base.mode == ProjectionBase.MUWEDGE:
            self.ells = ells or (0,2,4)
        else:
            self.ells = ells or self.input_base.projs
        projs = [ProjectionName((ProjectionName.MULTIPOLE,ell)) for ell in self.ells]
        data = DataVector(x=self.x,proj=projs)
        self.projection = ModelProjection(data,model_base=self.input_base,integration=integration)
        self.base = self.input_base.copy()
        if self.input_base.space == ProjectionBase.POWER:
            self.fftlog = PowerToCorrelation(self.x,ell=self.ells,q=q,lowring=False,xy=1)
            self.base.space = ProjectionBase.CORRELATION
        if self.input_base.space == ProjectionBase.CORRELATION:
            self.fftlog = CorrelationToPower(self.x,ell=self.ells,q=q,lowring=False,xy=1)
            self.base.space = ProjectionBase.POWER
        self.base.x = self.fftlog.y[0]
        self.base.mode = ProjectionBase.MULTIPOLE
        self.base.projs = self.ells

    def set_damping(self):
        x = self.x
        self.damping = 1.
        if self.input_base.space == ProjectionBase.POWER:
            self.damping = np.ones(x.size,dtype='f8')
            cutoff = 2.
            high = x>cutoff
            self.damping[high] *= np.exp(-(x[high]/cutoff-1.)**2)
            cutoff = 1e-4
            low = x<cutoff
            self.damping[low] *= np.exp(-(cutoff/x[low]-1.)**2)

    def eval(self, x, **kwargs):
        modelell = self.projection(self.input_model,concatenate=False,**kwargs)*self.damping
        modelell = self.fftlog(modelell)[-1].T
        return interpolate.interp1d(self.base.x,modelell,axis=0,kind='cubic',bounds_error=True,assume_sorted=True)(x)
