from scipy import special, interpolate

from pybird_dev import pybird

from cosmopipe.lib import utils, theory
from cosmopipe import section_names


class PyBird(object):

    def init_pybird(self):
        cache = dict(Nl=len(self.ells),kmax=self.options['kmax'],km=self.options['km'],optiresum=self.options['with_resum'] == 'opti',nd=self.nd)
        if cache == self._cache.get('init',{}):
            return
        self._cache['init'] = cache
        self.co = pybird.Common(halohalo=True,with_cf=False,with_time=True,exact_time=False,quintessence=False,with_tidal_alignments=False,nonequaltime=False,**cache)
        # TODO: PyBird: allow matrices to be passed on input
        self.nonlinear = pybird.NonLinear(load=False,save=False,co=self.co)
        self.resum = pybird.Resum(co=self.co)

    def setup(self):
        zeff = self.data_block[section_names.survey_geometry,'zeff']
        pklin = self.data_block[section_names.primordial_perturbations,'pk_callable']
        self.sigma8 = pklin.sigma8_z(zeff)
        klin,pklin = pklin.k,pklin(pklin.k,z=zeff)
        fo = self.data_block[section_names.primordial_cosmology,'cosmo'].get_fourier()
        self.growth_rate = fo.sigma8_z(zeff,of='theta_cb')/fo.sigma8_z(zeff,of='delta_cb')
        self.ells = [0,2,4]
        self.nd = self.data_block.get(section_names.data,'shotnoise',1e-4)
        self.init_pybird()

        cosmo = {'k11':klin,'P11':pklin,'f':self.growth_rate,'DA':1.,'H':1.}
        self.bird = pybird.Bird(cosmo,with_bias=True,with_stoch=self.options['with_stoch'],with_nlo_bias=self.options['with_nlo_bias'],co=self.co)
        self.nonlinear.PsCf(self.bird)

        self.required_params = ['b1', 'b2', 'b3', 'b4', 'cct', 'cr1']
        if len(self.ells) > 2:
            self.required_params += ['cr2']
        if self.options['with_stoch']:
            self.required_params += ['ce0','ce1','ce2']
        if self.options['with_nlo_bias']:
            self.required_params += ['bnlo']
        self.legendre = [special.legendre(ell) for ell in self.ells]

    def execute(self):
        pars = {}
        for name in self.required_params:
            pars[name] = self.data_block.get(section_names.galaxy_bias,name)
        fsig = self.data_block.get(section_names.galaxy_rsd,'fsig',self.growth_rate*self.sigma8)
        f = fsig/self.sigma8
        qpar = self.data_block.get(section_names.effect_ap,'qpar',1.)
        qperp = self.data_block.get(section_names.effect_ap,'qperp',1.)

        #if f != self._cache.get('f',None):
        self.bird.f = f
        self.bird.setPsCf(pars)
        if self.options['with_resum']: self.resum.Ps(self.bird)
        #self._cache['f'] = f
        pk_interp = interpolate.interp1d(self.co.k,self.bird.fullPs,axis=-1,kind='cubic',bounds_error=True,assume_sorted=True)

        def pk_mu_callable(k, mu, grid=True):
            pk = pk_interp(k)
            k,mu = utils.enforce_shape(k,mu,grid=grid)
            pk_mu = 0
            for p,leg in zip(pk,self.legendre): pk_mu += p*leg(mu)
            return pk_mu

        self.model = theory.EffectAP(pk_mu=pk_mu_callable)
        self.model.set_scaling(qpar=qpar,qperp=qperp)

        self.data_block[section_names.model,'y_callable'] = self.data_block[section_names.galaxy_power,'pk_mu_callable'] = self.model.pk_mu

    def cleanup(self):
        pass
