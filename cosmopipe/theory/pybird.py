from scipy import special, interpolate

from pybird import pybird

from cosmopipe.lib import utils, theory
from cosmopipe import section_names


class PyBird(object):

    def setup(self):
        zeff = self.data_block[section_names.survey_geometry,'zeff']
        pklin = self.data_block[section_names.primordial_perturbations,'pk_callable']
        self.sigma8 = pklin.sigma8_z(zeff)
        klin,pklin = pklin.k,pklin(pklin.k,z=self.zeff)
        fo = self.data_block[section_names.primordial_cosmology,'cosmo'].get_fourier()
        self.growth_rate = fo.sigma8_z(zeff,of='theta_cb')/fo.sigma8_z(zeff,of='delta_cb')
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
        fsig = self.data_block.get(section_names.galaxy_rsd,'fsig',self.growth_rate*self.sigma8)
        f = fsig/self.sigma8
        qpar = self.data_block.get(section_names.effect_ap,'qpar',1.)
        qperp = self.data_block.get(section_names.effect_ap,'qperp',1.)

        self.bird.f = f
        self.bird.setPsCf(pars)
        self.bird.setfullPs()
        self.resum.Ps(self.bird)

        def pk_mu_callable(k,mu):
            pk = interpolate.interp1d(self.co.k,self.bird.fullPs,axis=-1,kind='cubic',bounds_error=True,assume_sorted=True)(k)
            k,mu = utils.enforce_shape(k,mu)
            pk_mu = pk[0]*self.legendre[0](mu) + pk[1]*self.legendre[0](mu)
            return pk_mu

        self.model = theory.EffectAP(pk_mu=pk_mu_callable)
        self.model.set_scaling(qpar=qpar,qperp=qperp)

        self.data_block[section_names.model,'y_callable'] = self.data_block[section_names.galaxy_power,'pk_mu_callable'] = self.model.pk_mu

    def cleanup(self):
        pass
