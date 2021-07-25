from scipy import special, interpolate

from pybird_dev import pybird

from cosmopipe.lib import utils, theory
from cosmopipe.lib.theory.base import BaseModel, ProjectionBase, ModelCollection
from cosmopipe.lib.modules import ParameterizedModule
from cosmopipe import section_names


class PyBird(ParameterizedModule):

    def init_pybird(self):
        cache = dict(Nl=len(self.ells),kmax=self.kmax,km=self.km,optiresum=self.with_resum == 'opti',nd=self.nd,with_cf=self.with_correlation)
        newco = cache != self._cache.get('common',{})
        if newco:
            self._cache['common'] = cache
            self.co = pybird.Common(halohalo=True,with_time=True,exact_time=False,quintessence=False,with_tidal_alignments=False,nonequaltime=False,**cache)
            # TODO: PyBird: allow matrices to be passed on input
            self.nonlinear = pybird.NonLinear(load=False,save=False,co=self.co)
            #self.nonlinear = pybird.NonLinear(load=True,save=True,co=self.co)
            self.resum = pybird.Resum(co=self.co)
        cache = dict(nnlo_higher_derivative=self.with_nnlo_higher_derivative,with_nnlo_counterterm=self.with_nnlo_counterterm,with_cf=self.with_correlation)
        if newco or cache != self._cache.get('nnlo',{}):
            self._cache['nnlo'] = cache
            if self.with_nnlo_higher_derivative:
                self.nnlo_higher_derivative = pybird.NNLO_higher_derivative(self.co.k,with_cf=self.with_correlation,co=self.co)
            if self.with_nnlo_counterterm:
                self.nnlo_counterterm = pybird.NNLO_counterterm(co=self.co)

    def set_pklin(self):
        self.zeff = self.data_block[section_names.survey_selection,'zeff']
        pklin = self.data_block[section_names.primordial_perturbations,'pk_callable']
        self.sigma8 = pklin.sigma8_z(self.zeff)
        self.klin,self.pklin = pklin.k,pklin(pklin.k,z=self.zeff)
        fo = self.data_block[section_names.primordial_cosmology,'cosmo'].get_fourier()
        self.growth_rate = fo.sigma8_z(self.zeff,of='theta_cb')/fo.sigma8_z(self.zeff,of='delta_cb')

    def set_pknow(self):
        pklin = self.data_block[section_names.primordial_perturbations,'pk_callable'].to_1d(z=self.zeff)
        pknow_callable = PowerSpectrumBAOFilter(pklin,engine='wallish2018').smooth_pk_interpolator()

        def pknow_loglog(logk):
            return np.log(pknow_callable(logk,islogk=True))

        self.pknow_loglog = pknow_loglog
        self.pknow = pknow_callable(self.co.k)

    def setup(self):
        self.set_param_block()
        self.set_pklin()
        self.ells = [0,2,4]
        self.data_shotnoise = self.options.get('data_shotnoise',None)
        try:
            self.data_shotnoise = 1. * self.data_shotnoise
        except TypeError:
            self.data_shotnoise = self.data_block[section_names.data,'data_vector'].get(self.data_shotnoise,permissive=True)[0].attrs['shotnoise']
        self.nd = 1./self.data_shotnoise
        self.with_stoch = self.options.get('with_stoch',False)
        self.with_nnlo_higher_derivative = self.options.get('with_nnlo_higher_derivative',False)
        self.with_nnlo_counterterm = self.options.get('with_nnlo_counterterm',False)
        self.kmax = self.options.get('kmax',0.25)
        self.km = self.options.get('km',1.)
        self.with_resum = self.options.get('with_resum','opti')
        output = self.options.get_list('output',['power'])
        self.with_power = 'power' in output
        self.with_correlation = 'correlation' in output
        self.model_attrs = self.options.get_dict('model_attrs',{})
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

        model_collection = self.data_block.get(section_names.model,'collection',ModelCollection())
        if self.with_correlation:
            self.model_correlation = BaseModel(base=ProjectionBase(x=self.co.s,space=ProjectionBase.CORRELATION,mode=ProjectionBase.MULTIPOLE,projs=self.ells,wa_order=0,**self.model_attrs))
            model_collection.set(self.model_correlation)

        if self.with_power:
            self.model_power = BaseModel(base=ProjectionBase(x=self.co.k,space=ProjectionBase.POWER,mode=ProjectionBase.MULTIPOLE,projs=self.ells,wa_order=0,**self.model_attrs))
            model_collection.set(self.model_power)

        self.data_block[section_names.model,'collection'] = model_collection

    def execute(self):
        bias = {}
        for name in self.required_params:
            bias[name] = self.data_block.get(section_names.galaxy_bias,name)
        fsig = self.data_block.get(section_names.galaxy_rsd,'fsig',self.growth_rate*self.sigma8)
        f = fsig/self.sigma8

        #if f != self._cache.get('f',None):
        self.bird.f = f
        self.bird.setPsCf(bias)
        if self.with_resum:
            if self.with_correlation:
                self.resum.PsCf(self.bird)
            else:
                self.resum.Ps(self.bird)
        if self.with_nnlo_higher_derivative:
            self.bird_now.setPsCf(self.bias)
            bias_local = bias #.copy() # we remove the counterterms and the stochastic terms, if any.
            for name in ['cct','cr1','cr2','ce0','ce1','ce2']:
                bias_local[name] = 0.
            if self.with_correlation:
                self.bird_now.setreduceCflb(bias_local) # PS included
            else:
                self.bird_now.setreducePslb(bias_local)
            bnnlo = np.array([bias_local['bnnlo{:d}'.format(ell)] for ell in self.ells])
            if self.with_correlation:
                correlation_nnlo = self.nnlo_higher_derivative.Cf(self.bird_now) # k^2 P1Loop
                correlation_nnlo = np.einsum('l,lx->lx',bnnlo,correlation_nnlo)
            if self.with_power:
                power_nnlo = self.nnlo_higher_derivative.Ps(self.bird_now) # k^2 P1Loop
                power_nnlo = np.einsum('l,lx->lx',bnnlo,power_nnlo)
        if self.with_nnlo_counterterm:
            if self.with_correlation:
                self.nnlo_counterterm.Cf(self.bird,self.pknow_loglog)
            if self.with_power:
                self.nnlo_counterterm.Ps(self.bird,self.pknow_loglog)
            # self.bird.Pnnlo = self.co.k**4 * self.pknow # equivalent to the commented line above, just make it less stupid
        #self._cache['f'] = f
        if self.with_power:
            power = self.bird.fullPs + self.data_shotnoise
            if self.with_nnlo_higher_derivative:
                power += power_nnlo
            self.model_power.eval = interpolate.interp1d(self.co.k,power.T,axis=0,kind='cubic',bounds_error=True,assume_sorted=True)

        if self.with_correlation:
            correlation = self.bird.fullCf
            if self.with_nnlo_higher_derivative:
                correlation += correlation_nnlo
            self.model_correlation.eval = interpolate.interp1d(self.co.s,correlation.T,axis=0,kind='cubic',bounds_error=True,assume_sorted=True)

    def cleanup(self):
        pass
