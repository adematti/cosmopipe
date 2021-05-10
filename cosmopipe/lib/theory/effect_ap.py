import numpy as np

from cosmopipe.lib import utils
from cosmopipe.lib.utils import BaseClass


class EffectAP(BaseClass):

    def __init__(self, pk_mu=None):
        self.input_pk_mu = pk_mu
        self.set_scaling()

    def set_scaling(self, qpar=1, qperp=1):
        self.qpar = qpar
        self.qperp = qperp
        self.qap = qpar/qperp
        self.qiso = (self.qperp**2*self.qpar)**(1./3.)

    def kmu_scaling(self, k, mu, grid=True):
        factor_ap = np.sqrt(1+mu**2*(1./self.qap**2-1))
        k,mu = utils.enforce_shape(k,mu,grid=grid)
        # Beutler 2016 (arXiv: 1607.03150v1) eq 44
        kap = k/self.qperp*factor_ap
        # Beutler 2016 (arXiv: 1607.03150v1) eq 45
        muap = mu/self.qap/factor_ap
        return kap,muap

    def pk_mu(self, k, mu=0., grid=True, **kwargs):
        kap, muap = self.kmu_scaling(k,mu,grid=grid)
        return 1./self.qiso**3*self.input_pk_mu(k=kap,mu=muap,grid=False,**kwargs)


class IsotropicScaling(BaseClass):

    def __init__(self, pk=None, pivot=1./3.):
        self.input_pk = pk
        self.pivot = pivot
        self.set_scaling()

    def set_scaling(self, qpar=1., qperp=1.):
        self.qiso = qpar**self.pivot*qperp**(1.-self.pivot)
        self.qap = qpar/qperp

    def anisotropic_scaling(self):
        return self.qap**(1.-self.pivot), self.qap**(-self.pivot)

    def k_scaling(self, k):
        return k/self.qiso

    def pk(self, k, **kwargs):
        return 1/self.qiso**3*self.input_pk(k=self.k_scaling(k),**kwargs)
