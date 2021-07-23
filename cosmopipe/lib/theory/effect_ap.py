import numpy as np

from cosmopipe.lib import utils
from cosmopipe.lib.utils import BaseClass

from .base import ProjectionBase, BaseModel
from .integration import MultipoleExpansion


class AnisotropicScaling(BaseModel):

    def __init__(self, model=None, base=None):
        self.input_model = model
        self.base = self.input_base = base if base is not None else self.input_model.base
        if self.input_base.mode == ProjectionBase.MULTIPOLE:
            self.multipole_expansion = MultipoleExpansion(self.input_model,ells=self.base.projs)
            self.base = self.input_base.copy()
            self.base.mode = ProjectionBase.MUWEDGE
        self.set_scaling()

    def set_scaling(self, qpar=1, qperp=1):
        self.qpar = qpar
        self.qperp = qperp
        self.qap = qpar/qperp
        self.qiso = (self.qperp**2*self.qpar)**(1./3.)

    def kmu_scaling(self, k, mu, grid=True):
        factor_ap = np.sqrt(1 + mu**2*(1./self.qap**2-1))
        k,mu = utils.enforce_shape(k,mu,grid=grid)
        # Beutler 2016 (arXiv: 1607.03150) eq 44
        kap = k/self.qperp*factor_ap
        # Beutler 2016 (arXiv: 1607.03150) eq 45
        muap = mu/self.qap/factor_ap
        return kap, muap

    def smu_scaling(self, s, mu, grid=True):
        factor_ap = np.sqrt(mu**2*(self.qap**2-1) + 1)
        # Hou 2018 (arXiv: 2007.08998) eq 8
        sap = s*self.qperp*factor_ap
        muap = mu*self.qap/factor_ap
        return sap, muap

    def eval(self, x, mu=0., grid=True, **kwargs):
        if self.input_base.mode == ProjectionBase.MULTIPOLE:
            self.multipole_expansion.input_model = self.input_model
            input_model = self.multipole_expansion
        else:
            input_model = self.input_model
        if self.base.space == ProjectionBase.CORRELATION:
            sap, muap = self.smu_scaling(x,mu,grid=grid)
            return input_model(sap,mu=muap,grid=False,**kwargs)
        kap, muap = self.kmu_scaling(x,mu,grid=grid)
        return 1./self.qiso**3*input_model(kap,mu=muap,grid=False,**kwargs)



class IsotropicScaling(BaseModel):

    def __init__(self, model=None, base=None, pivot=1./3.):
        self.input_model = model
        self.base = base if base is not None else self.input_model.base
        self.pivot = pivot
        self.set_scaling()

    def set_scaling(self, qpar=1., qperp=1.):
        self.qiso = qpar**self.pivot*qperp**(1.-self.pivot)
        self.qap = qpar/qperp

    def anisotropic_scaling(self):
        return self.qap**(1.-self.pivot), self.qap**(-self.pivot)

    def k_scaling(self, k):
        return k/self.qiso

    def s_scaling(self, s):
        return s/self.qiso

    def eval(self, x, **kwargs):
        if self.base.space == ProjectionBase.CORRELATION:
            return self.input_model(self.s_scaling(x),**kwargs)
        return 1/self.qiso**3*self.input_model(self.k_scaling(x),**kwargs)
