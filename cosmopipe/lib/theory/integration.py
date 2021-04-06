import numpy as np

from cosmopipe.lib.utils import BaseClass


def weights_trapz(x):
    return np.concatenate([[x[1]-x[0]],x[2:]-x[:-2],[x[-1]-x[-2]]])/2.


class BaseMultipolesIntegration(BaseClass):

    def __init__(self, mu=100, ells=(0,2,4)):
        self.mu = mu
        self.ells = ells
        self.set_mu_weights()

    def __call__(self, array):
        return np.sum(array*self.muw[:,None,:],axis=-1)


# TODO: implement legendre integration
class TrapzMultipolesIntegration(BaseMultipolesIntegration):

    def set_mu_weights(self):
        if np.isscalar(self.mu):
            self.mu = np.linspace(0.,1.,self.mu)
        muw_trapz = weights_trapz(self.mu)
        from scipy import special
        self.muw = np.array([muw_trapz*(2*ell+1.)*special.legendre(ell)(self.mu) for ell in self.ells])/(self.mu[-1]-self.mu[0])


def MultipolesIntegration(multipoles_integration=None):
    default = {'mu':100, 'ells':(0,2,4)}
    if multipoles_integration is None:
        multipoles_integration = {}
    if isinstance(multipoles_integration,dict):
        return TrapzMultipolesIntegration(**{**default,**multipoles_integration})
    return multipoles_integration
