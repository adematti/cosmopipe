import numpy as np
from scipy import special

from cosmopipe.lib import utils
from cosmopipe.lib.utils import BaseClass


def weights_trapz(x):
    return np.concatenate([[x[1]-x[0]],x[2:]-x[:-2],[x[-1]-x[-2]]])/2.


class BaseMultipoleIntegration(BaseClass):

    def __init__(self, mu=100, ells=(0,2,4), sym=False):
        self.mu = mu
        if np.ndim(self.mu) == 0:
            self.mu = np.linspace(0. if sym else -1.,1.,self.mu+1)
        self.ells = ells
        self.set_mu_weights()

    def __call__(self, array):
        return np.sum(array*self.weights[:,None,:],axis=-1)


# TODO: implement gauss-legendre integration
class TrapzMultipoleIntegration(BaseMultipoleIntegration):

    def set_mu_weights(self):
        muw_trapz = weights_trapz(self.mu)
        from scipy import special
        self.weights = np.array([muw_trapz*(2*ell+1.)*special.legendre(ell)(self.mu) for ell in self.ells])/(self.mu[-1]-self.mu[0])


class BaseMuWedgeIntegration(BaseClass):

    def __init__(self, mu=100, muwedges=3):
        self.mu = mu
        if np.ndim(self.mu) == 0:
            if np.ndim(muwedges) == 0:
                muwedges = np.linspace(0.,1.,muwedges+1)
                muwedges = zip(muwedges[:-1],muwedges[1:])
            self.mu = [np.linspace(*muwedge,mu//len(muwedges)) for muwedge in muwedges]
        self.set_mu_weights()

    def __call__(self, array):
        return np.sum(array*self.weights[:,None,:],axis=-1)


# TODO: implement gauss-legendre integration
class TrapzMuWedgeIntegration(BaseMuWedgeIntegration):

    def set_mu_weights(self):
        self.weights = np.array([weights_trapz(mu)/(mu[-1]-mu[0]) for mu in self.mu])


class MultipoleExpansion(BaseClass):

    def __init__(self, input_model=None, ells=(0,2,4)):
        self.legendre = [special.legendre(ell) for ell in ells]
        self.input_model = input_model

    def __call__(self, x, mu, grid=True, **kwargs):
        y = self.input_model(x)
        x,mu = utils.enforce_shape(x,mu,grid=grid)
        toret = 0
        for y_,leg in zip(y,self.legendre): toret += y_*leg(mu)
        return toret


class MultipoleToMultipole(BaseClass):

    def __init__(self, ellsin=(0,2,4), ells=(0,2,4)):
        self.weights = np.zeros((len(ellsin),len(ells)),dtype='f8')
        ells = np.array(ells)
        for illin,ellin in enumerate(ellsin):
            self.weights[illin,ells == ellin] = 1.

    def __call__(self, array):
        return array.dot(self.weights)


class MultipoleToMuWedge(BaseClass):

    def __init__(self, ellsin=(0,2,4), muwedges=3):
        if np.ndim(muwedges) == 0:
            muwedges = np.linspace(0.,1.,muwedges+1)
            muwedges = zip(muwedges[:-1],muwedges[1:])
        if np.ndim(muwedges[0]) == 0:
            muwedges = [muwedges]
        integlegendre = [special.legendre(ell).integ() for ell in ellsin]
        mulow,muup = np.array([muwedge[0] for muwedge in muwedges]),np.array([muwedge[-1] for muwedge in muwedges])
        muwidth = muup - mulow
        self.weights = np.array([(poly(muup) - poly(mulow))/muwidth for poly in integlegendre])

    def __call__(self, array):
        return array.dot(self.weights)



def MultipoleIntegration(integration=None):
    default = {'mu':100,'ells':(0,2,4)}
    if integration is None:
        integration = {}
    if isinstance(integration,dict):
        return TrapzMultipoleIntegration(**{**default,**integration})
    return integration


def MuWedgeIntegration(integration=None):
    default = {'mu':100,'muwedges':3}
    if integration is None:
        integration = {}
    if isinstance(integration,dict):
        return TrapzMuWedgeIntegration(**{**default,**integration})
    return integration
