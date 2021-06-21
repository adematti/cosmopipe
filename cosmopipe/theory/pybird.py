from scipy import special, interpolate

from pybird_dev import pybird

from cosmopipe import section_names
from cosmopipe.lib import utils, theory
from cosmopipe.lib.theory.projection import ProjectionBase
from cosmopipe.lib.theory.integration import MultipoleExpansion


class PyBird(object):

    def init_pybird(self):
        cache = dict(Nl=len(self.ells),kmax=self.kmax,km=self.km,optiresum=self.with_resum == 'opti',nd=self.nd)
        if cache != self._cache.get('common',{}):
            self._cache['common'] = cache
            self.co = pybird.Common(halohalo=True,with_cf=False,with_time=True,exact_time=False,quintessence=False,with_tidal_alignments=False,nonequaltime=False,**cache)
            # TODO: PyBird: allow matrices to be passed on input
            self.nonlinear = pybird.NonLinear(load=False,save=False,co=self.co)
            #self.nonlinear = pybird.NonLinear(load=True,save=True,co=self.co)
            self.resum = pybird.Resum(co=self.co)
        cache = dict(nnlo_higher_derivative=self.with_nnlo_higher_derivative,with_nnlo_counterterm=self.with_nnlo_counterterm)
        if cache != self._cache.get('nnlo',{}):
            self._cache['nnlo'] = cache
            if self.with_nnlo_higher_derivative:
                self.nnlo_higher_derivative = pybird.NNLO_higher_derivative(self.co.k,with_cf=False,co=self.co)
            if self.with_nnlo_counterterm:
                self.nnlo_counterterm = pybird.NNLO_counterterm(co=self.co)

    def set_pklin(self):
        self.zeff = self.data_block[section_names.survey_geometry,'zeff']
        pklin = self.data_block[section_names.primordial_perturbations,'pk_callable']
        self.sigma8 = pklin.sigma8_z(self.zeff)
        self.klin,self.pklin = pklin.k,pklin(pklin.k,z=self.zeff)
        fo = self.data_block[section_names.primordial_cosmology,'cosmo'].get_fourier()
        self.growth_rate = fo.sigma8_z(self.zeff,of='theta_cb')/fo.sigma8_z(self.zeff,of='delta_cb')

    def set_pknow(self):
        pklin = self.data_block[section_names.primordial_perturbations,'pk_callable'].to_1d(z=self.zeff)
        self.pknow = PowerSpectrumBAOFilter(pklin,engine='wallish2018').smooth_pk_interpolator()(self.co.k)

    def setup(self):
        self.set_pklin()
        self.ells = [0,2,4]
        self.nd = self.data_block.get(section_names.data,'shotnoise',1e-4)
        self.with_stoch = self.options.get('with_stoch',False)
        self.with_nnlo_higher_derivative = self.options.get('with_nnlo_higher_derivative',False)
        self.with_nnlo_counterterm = self.options.get('with_nnlo_counterterm',False)
        self.kmax = self.options.get('kmax',0.25)
        self.km = self.options.get('km',1.)
        self.with_resum = self.options.get('with_resum','opti')
        self.init_pybird()

        cosmo = {'k11':self.klin,'P11':self.pklin,'f':self.growth_rate,'DA':1.,'H':1.}
        self.bird = pybird.Bird(cosmo,with_bias=True,with_stoch=self.with_stoch,with_nnlo_counterterm=self.with_nnlo_counterterm,co=self.co)
        self.nonlinear.PsCf(self.bird)

        if self.with_nnlo_higher_derivative:
            self.bird_now = deepcopy(self.bird)
            self.bird_now.Pin = self.pknow
            self.nonlinear.PsCf(self.bird_now)

        self.required_params = ['b1', 'b2', 'b3', 'b4', 'cct', 'cr1', 'cr2']
        if self.with_stoch:
            self.required_params += ['ce0','ce1','ce2']
        if self.with_nnlo_higher_derivative:
            self.required_params += ['bnnlo0','bnnlo2']
            if len(self.ells) > 2: self.required_params += ['bnnlo4']
        if self.with_nnlo_counterterm:
            self.required_params += ['cnnlo0','cnnlo2']
            if len(self.ells) > 2: self.required_params += ['cnnlo4']
        if self.with_nnlo_higher_derivative or self.with_nnlo_counterterm:
            self.set_pknow()
        self.multipole_expansion = MultipoleExpansion(ells=self.ells)
        self.data_block[section_names.model,'y_base'] = ProjectionBase('muwedge')

    def execute(self):
        bias = {}
        for name in self.required_params:
            bias[name] = self.data_block.get(section_names.galaxy_bias,name)
        fsig = self.data_block.get(section_names.galaxy_rsd,'fsig',self.growth_rate*self.sigma8)
        f = fsig/self.sigma8
        qpar = self.data_block.get(section_names.effect_ap,'qpar',1.)
        qperp = self.data_block.get(section_names.effect_ap,'qperp',1.)

        #if f != self._cache.get('f',None):
        self.bird.f = f
        self.bird.setPsCf(bias)
        if self.with_resum: self.resum.Ps(self.bird)
        if self.with_nnlo_higher_derivative:
            self.bird_now.setPsCf(self.bias)
            bias_local = bias #.copy() # we remove the counterterms and the stochastic terms, if any.
            for name in ['cct','cr1','cr2','ce0','ce1','ce2']:
                bias_local[name] = 0.
            self.bird_now.setreducePslb(bias_local)
            bnnlo = np.array([bias_local['bnnlo{:d}'.format(ell)] for ell in self.ells])
            nnlo = self.nnlo_higher_derivative.Ps(self.bird_now) # k^2 P1Loop
            nnlo = np.einsum('l,lx->lx',bnnlo,nnlo)
        if self.with_nnlo_counterterm:
            # self.nnlo_counterterm.Ps(self.bird,self.pknow)
            self.bird.Pnnlo = self.co.k**4 * self.pknow # equivalent to the commented line above
        #self._cache['f'] = f
        pk = self.bird.fullPs
        if self.with_nnlo_higher_derivative:
            pk += nnlo
        pk_interp = interpolate.interp1d(self.co.k,pk,axis=-1,kind='cubic',bounds_error=True,assume_sorted=True)
        self.multipole_expansion.input_fun = pk_interp

        self.model = theory.EffectAP(pk_mu=self.multipole_expansion)
        self.model.set_scaling(qpar=qpar,qperp=qperp)

        self.data_block[section_names.model,'x'] = self.co.k
        self.data_block[section_names.model,'y_callable'] = self.data_block[section_names.galaxy_power,'pk_mu_callable'] = self.model.pk_mu

    def cleanup(self):
        pass
