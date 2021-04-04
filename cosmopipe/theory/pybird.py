from scipy import special, interpolate

from pybird import pybird

from cosmopipe.lib import theory
from cosmopipe import section_names


class PyBird(object):

    def setup(self):
        pklin = self.data_block[section_names.linear_perturbations,'pk_callable']
        klin,pklin = pklin['k'],pklin['pk']
        self.growth_rate = self.data_block[section_names.background,'growth_rate']
        kwargs = {}
        kwargs['optiresum'] = self.options.get_bool('optiresum',True)
        self.co = pybird.Common(optiresum=kwargs['optiresum'])
        #TODO: PyBird: allow matrices to be passed on input
        self.nonlinear = pybird.NonLinear(load=True,save=True,co=self.co)
        self.resum = pybird.Resum(co=self.co)
        self.bird = pybird.Bird(klin,pklin,f=self.growth_rate,which='all',co=self.co)
        self.nonlinear.PsCf(self.bird)
        #self.required_params = ['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7']
        self.required_params = ['b1', 'b2', 'b3', 'b4', 'ct', 'cr1', 'cr2']
        self.legendre = [special.legendre(ell) for ell in [0,2]]

    def execute(self):
        pars = []
        for par in self.required_params:
            pars.append(self.data_block.get(section_names.galaxy_bias,par))
        f = self.data_block.get(section_names.galaxy_rsd,'f',self.growth_rate)
        qpar = self.data_block.get(section_names.effect_ap,'qpar',1.)
        qperp = self.data_block.get(section_names.effect_ap,'qperp',1.)

        self.bird.f = f
        self.bird.setPsCf(pars)
        self.bird.setfullPs()
        self.resum.Ps(self.bird)

        def pk_mu_callable(k,mu):
            pk = interpolate.interp1d(self.co.k,self.bird.fullPs,axis=-1,kind='cubic',bounds_error=True,assume_sorted=True)(k)
            pk_mu = pk[0]*self.legendre[0](mu)[:,None] + pk[1]*self.legendre[1](mu)[:,None]
            return pk_mu.T

        self.data_block[section_names.model,'y_callable'] = self.data_block[section_names.galaxy_power,'pk_mu_callable'] = pk_mu_callable

    def cleanup(self):
        pass
