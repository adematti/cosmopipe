import numpy as np

from cosmopipe.lib.utils import BaseClass


class EffectAP(BaseClass):

    def __init__(self, pk_mu):
        self.input_pk_mu = pk_mu

    @staticmethod
    def kmu_scaling(k, mu, qpar=1, qperp=1):
        F = qpar/qperp
        factor_ap = np.sqrt(1+mu**2*(1./F**2-1))
        if not np.isscalar(mu):
            k = k[:,None]
        # Beutler 2016 (arXiv: 1607.03150v1) eq 44
        kap = k/qperp*factor_ap
        # Beutler 2016 (arXiv: 1607.03150v1) eq 45
        muap = mu/F/factor_ap
        jacob = 1./(qperp**2*qpar)
        return jacob,kap,muap

    def pk_mu(self, k, mu=0., qpar=1., qperp=1.,**kwargs):
        jacob,kap,muap = self.kmu_scaling(k,mu,qpar=qpar,qperp=qperp)
        return jacob*self.input_pk_mu(k=k,mu=mu,**kwargs)
