import numpy as np
from scipy import interpolate

from cosmoprimo import Cosmology, PowerSpectrumBAOFilter

from cosmopipe.lib.theory.base import BaseModel, ProjectionBase, ModelCollection
from cosmopipe.lib.parameter import ParamName
from cosmopipe.lib.modules import ParameterizedModule
from cosmopipe import section_names


class Velocileptors(ParameterizedModule):

    def get_theory_options(self):
        options = self._default_options.copy()
        for name,value in self._default_options.items():
            options[name] = self.options.get(name,value)
        options['threads'] = options.pop('nthreads')
        self.model_attrs = self.options.get_dict('model_attrs',{})
        self.data_shotnoise = self.options.get('data_shotnoise',None)
        try:
            self.data_shotnoise = 1. * self.data_shotnoise
        except TypeError:
            self.data_shotnoise = self.data_block[section_names.data,'data_vector'].get(self.data_shotnoise,permissive=True)[0].attrs['shotnoise']
        return options

    def set_model(self, space=ProjectionBase.POWER, mode=ProjectionBase.MUWEDGE, projs=None):
        include = [ParamName(section_names.galaxy_bias,name) for name in self.required_params if name not in self.default_required_params]
        include += [ParamName(section_names.galaxy_bias,name) for name in self.optional_params]
        self.set_param_block(include=include)
        self.model = BaseModel(base=ProjectionBase(x=self.theory.kv if space == ProjectionBase.POWER else self.theory.rint,space=space,mode=mode,projs=projs,**self.model_attrs))
        self.data_block[section_names.model,'collection'] = self.data_block.get(section_names.model,'collection',[]) + ModelCollection([self.model])

    def set_pklin(self):
        self.zeff = self.data_block[section_names.survey_selection,'zeff']
        pklin = self.data_block[section_names.primordial_perturbations,'pk_callable']
        self.sigma8 = pklin.sigma8_z(self.zeff)
        self.klin,self.pklin = pklin.k,pklin(pklin.k,z=self.zeff)
        fo = self.data_block[section_names.primordial_cosmology,'cosmo'].get_fourier()
        self.growth_rate = fo.sigma8_z(self.zeff,of='theta_cb')/fo.sigma8_z(self.zeff,of='delta_cb')

    def set_pknow(self):
        pklin = self.data_block[section_names.primordial_perturbations,'pk_callable'].to_1d(z=self.zeff)
        self.pknow = PowerSpectrumBAOFilter(pklin,engine='wallish2018').smooth_pk_interpolator()(self.klin)

    def get_model_callable(self, pars, f, **kwargs):

        def model_callable(k, mu, grid=True):
            if grid:
                tmp = np.array([self.theory.compute_redshift_space_power_at_mu(pars,f,mu_,apar=1.,aperp=1.,**kwargs)[-1] for mu_ in mu]).T
                toret = interpolate.interp1d(self.theory.kv,tmp,kind='cubic',axis=0,copy=False,bounds_error=True,assume_sorted=True)(k)
            else:
                toret = np.empty(k.shape,dtype=k.dtype)
                for imu,mu_ in enumerate(mu):
                    tmp = self.theory.compute_redshift_space_power_at_mu(pars,f,mu_,apar=1.,aperp=1.,**kwargs)[-1]
                    toret[:,imu] = interpolate.interp1d(self.theory.kv,tmp,kind='cubic',axis=0,copy=False,bounds_error=True,assume_sorted=True)(k[:,imu])
            return toret + self.data_shotnoise

        return model_callable

    def execute(self):
        pars = []
        for par in self.required_params:
            pars.append(self.data_block.get(section_names.galaxy_bias,par))
        for par,value in self.default_required_params.items():
            pars.append(value)
        opts = {}
        for par in self.optional_params:
            opts[par] = self.data_block.get(section_names.galaxy_bias,par,self.optional_params[par])
        fsig = self.data_block.get(section_names.galaxy_rsd,'fsig',self.growth_rate*self.sigma8)
        f = fsig/self.sigma8

        self.model.eval = self.get_model_callable(pars,f,**opts,**self.optional_kw)
        self.data_block[section_names.model,'collection'] = self.data_block.get(section_names.model,'collection',[]) + ModelCollection([self.model])

    def cleanup(self):
        pass


class EPTMoments(Velocileptors):

    _default_options = dict(rbao=110,kmin=1e-2,kmax=0.5,nk=100,beyond_gauss=True,one_loop=True,shear=True,third_order=True,cutoff=20,jn=5,N=4000,nthreads=1,extrap_min=-4,extrap_max=3,import_wisdom=False)

    def setup(self):
        reduced = self.options.get('reduced',True)
        options = self.get_theory_options()
        self.set_pklin()
        self.set_pknow()
        from velocileptors.EPT.moment_expansion_fftw import MomentExpansion
        self.theory = MomentExpansion(self.klin,self.pklin,pnw=self.pknow,**options)
        if options['beyond_gauss']:
            if reduced:
                self.required_params = ['b1', 'b2', 'bs', 'b3', 'alpha0', 'alpha2', 'alpha4', 'alpha6', 'sn', 'sn2', 'sn4']
            else:
                self.required_params = ['b1', 'b2', 'bs', 'b3', 'alpha', 'alphav', 'alpha_s0', 'alpha_s2', 'alpha_g1',\
                                        'alpha_g3', 'alpha_k2', 'sn', 'sv', 'sigma0', 'stoch_k0']
        else:
            if reduced:
                self.required_params = ['b1', 'b2', 'bs', 'b3', 'alpha0', 'alpha2', 'alpha4', 'sn', 'sn2']
            else:
                self.required_params = ['b1', 'b2', 'bs', 'b3', 'alpha', 'alphav', 'alpha_s0', 'alpha_s2', 'sn', 'sv', 'sigma0']
        self.default_required_params = {}
        self.optional_params = dict(counterterm_c3=0.)
        self.optional_kw = dict(beyond_gauss=options['beyond_gauss'],reduced=reduced)
        self.set_model()


class EPTFull(Velocileptors):

    #_default_options = dict(rbao=110,kmin=1e-2,kmax=0.5,nk=100,sbao=None,beyond_gauss=True,one_loop=True,shear=True,third_order=True,cutoff=20,jn=5,N=4000,nthreads=1,extrap_min=-4,extrap_max=3,import_wisdom=False)
    _default_options = dict(rbao=110,kmin=1e-2,kmax=0.5,nk=100,sbao=None,beyond_gauss=True,one_loop=True,shear=True,cutoff=20,jn=5,N=4000,nthreads=1,extrap_min=-4,extrap_max=3,import_wisdom=False)

    def setup(self):
        reduced = self.options.get('reduced',True)
        options = self.get_theory_options()
        self.set_pklin()
        self.set_pknow()
        from velocileptors.EPT.ept_fullresum_fftw import REPT
        self.theory = REPT(self.klin,self.pklin,pnw=self.pknow,**options)
        self.required_params = ['b1', 'b2', 'bs', 'b3', 'alpha0', 'alpha2', 'alpha4', 'alpha6', 'sn', 'sn2', 'sn4']
        self.default_required_params = {}
        self.optional_params = dict(bFoG=0.)
        self.optional_kw = dict()
        self.set_model()


class LPTMoments(Velocileptors):

    _default_options = dict(kmin=5e-3,kmax=0.3,nk=50,beyond_gauss=False,one_loop=True,shear=True,third_order=True,cutoff=10,jn=5,N=2000,nthreads=None,extrap_min=-5,extrap_max=3,import_wisdom=False)

    def setup(self):
        reduced = self.options.get_bool('reduced',True)
        options = self.get_theory_options()
        self.set_pklin()
        from velocileptors.LPT.moment_expansion_fftw import MomentExpansion
        self.theory = MomentExpansion(self.klin,self.pklin,**options)
        if options['beyond_gauss']:
            if reduced:
                self.required_params = ['b1', 'b2', 'bs', 'b3', 'alpha0', 'alpha2', 'alpha4', 'alpha6', 'sn', 'sn2', 'sn4']
            else:
                self.required_params = ['b1', 'b2', 'bs', 'b3', 'alpha', 'alpha_v', 'alpha_s0', 'alpha_s2', 'alpha_g1',\
                                        'alpha_g3', 'alpha_k2', 'sn', 'sv', 'sigma0_stoch', 'sn4']
        else:
            if reduced:
                self.required_params = ['b1', 'b2', 'bs', 'b3', 'alpha0', 'alpha2', 'alpha4', 'sn', 'sn2']
            else:
                self.required_params = ['b1', 'b2', 'bs', 'b3', 'alpha', 'alpha_v', 'alpha_s0', 'alpha_s2', 'sn', 'sv', 'sigma0_stoch']
        self.default_required_params = {}
        if not options['third_order']:
            self.default_required_params['b3'] = 0.
            if not options['shear']:
                self.default_required_params['bs'] = 0.
        self.optional_params = dict(counterterm_c3=0.)
        self.optional_kw = dict(ngauss=4,reduced=reduced)
        self.set_model()


class LPTFourierStreaming(Velocileptors):

    _default_options = dict(kmin=1e-3,kmax=3,nk=100,beyond_gauss=False,one_loop=True,shear=True,third_order=True,cutoff=10,jn=5,N=2000,nthreads=None,extrap_min=-5,extrap_max=3,import_wisdom=False)

    def setup(self):
        # jn = 5
        options = self.get_theory_options()
        self.set_pklin()
        from velocileptors.LPT.fourier_streaming_model_fftw import FourierStreamingModel
        self.theory = FourierStreamingModel(self.klin,self.pklin,**options)
        # b3 if in third order, bs if shear bias
        self.required_params = ['b1', 'b2', 'bs', 'b3','alpha', 'alpha_v', 'alpha_s0', 'alpha_s2', 'sn', 'sv', 'sigma0_stoch']
        self.default_required_params = {}
        if not options['third_order']:
            self.default_required_params['b3'] = 0.
            if not options['shear']:
                self.default_required_params['bs'] = 0.
        self.optional_params = dict(counterterm_c3=0)
        self.optional_kw = dict()
        self.set_model()


class LPTGaussianStreaming(Velocileptors):

    #_default_options = dict(kmin=1e-3,kmax=3,nk=200,jn=10,cutoff=20,beyond_gauss=False,one_loop=True,shear=True,third_order=True,N=2000,nthreads=None,extrap_min=-5,extrap_max=3,import_wisdom=False)
    _default_options = dict(kmin=1e-3,kmax=3,nk=200,jn=10,cutoff=20,beyond_gauss=False,one_loop=True,shear=True,N=2000,nthreads=None,extrap_min=-5,extrap_max=3,import_wisdom=False)

    def setup(self):
        # kmin = 3e-3, kmax=0.5, nk = 100, kswitch=1e-2, jn = 5, cutoff=20
        options = self.get_theory_options()
        self.set_pklin()
        from velocileptors.LPT.gaussian_streaming_model_fftw import GaussianStreamingModel
        self.theory = GaussianStreamingModel(self.klin,self.pklin,**options)
        # alpha_s0 and alpha_s2 to be zeros
        self.required_params = ['b1', 'b2', 'bs', 'b3', 'alpha', 'alpha_v', 'alpha_s0', 'alpha_s2', 's2FoG']
        self.default_required_params = {}
        if not options.get('third_order',True):
            self.default_required_params['b3'] = 0.
            if not options['shear']:
                self.default_required_params['bs'] = 0.
        self.optional_params = dict()
        self.optional_kw = dict(rwidth=100,Nint=10000,ngauss=4,update_cumulants=False)
        self.set_model(space=ProjectionBase.CORRELATION,mode=ProjectionBase.MULTIPOLE,projs=(0,2,4))

    def get_model_callable(self, pars, f, **kwargs):

        def model_callable(s):
            xi = []
            for ss in s:
                xi0,xi2,xi4 = self.theory.compute_xi_ell(ss,f,*pars,apar=1.,aperp=1.)
                xi.append([xi0,xi2,xi4])
            return np.array(xi)

        return model_callable


class LPTDirect(Velocileptors):

    _default_options = dict(third_order=True,shear=True,one_loop=True,kIR=None,cutoff=10,jn=5,N=2000,nthreads=None,extrap_min=-5,extrap_max=3)

    def setup(self):
        # kmin = 3e-3, kmax=0.5, nk = 100, kswitch=1e-2, jn = 5, cutoff=20
        options = self.get_theory_options()
        output = self.options.get_list('output',['power'])
        self.with_power = 'power' in output
        self.with_correlation = 'correlation' in output
        self.set_pklin()
        from velocileptors.LPT.lpt_rsd_fftw import LPT_RSD
        self.theory = LPT_RSD(self.klin,self.pklin,**options)
        # alpha_s0 and alpha_s2 to be zeros
        self.required_params = ['b1', 'b2', 'bs', 'b3', 'alpha0', 'alpha2', 'alpha4', 'alpha6', 'sn', 'sn2', 'sn4']
        self.default_required_params = {}
        if not options['third_order']:
            self.default_required_params['b3'] = 0.
            if not options['shear']:
                self.default_required_params['bs'] = 0.
        self.optional_params = dict()
        self.optional_kw = dict(ngauss=3,kv=None,kmin=5e-3 if self.with_correlation else 1e-2,kmax=1.0 if self.with_correlation else 0.25,nk=60 if self.with_correlation else 50,nmax=4)
        if self.with_correlation:
            self.set_model(space=ProjectionBase.CORRELATION,mode=ProjectionBase.MULTIPOLE,projs=(0,2,4),**self.model_attrs)
            self.model_correlation = self.model
        if self.with_power:
            self.set_model(space=ProjectionBase.POWER,mode=ProjectionBase.MULTIPOLE,projs=(0,2,4),**self.model_attrs)
            self.model_power = self.model

    def get_correlation_callable(self, pars, **kwargs):

        def model_callable(s):
            xiells = lpt.combine_bias_terms_xiell(pars,**kwargs)
            return np.array([interpolate.interp1d(xiell[0],xiell[1],kind='cubic',axis=0,copy=False,bounds_error=True,assume_sorted=True)(s) for xiell in xiells])

        return model_callable

    def get_power_callable(self, pars, **kwargs):

        def model_callable(k):
            kl, p0, p2, p4 = self.theory.combine_bias_terms_pkell(pars,**kwargs)
            p0 += self.data_shotnoise
            pkells = np.array([p0,p2,p4]).T
            return interpolate.interp1d(kl,pkells,kind='cubic',axis=0,copy=False,bounds_error=True,assume_sorted=True)(k)

        return model_callable

    def execute(self):
        pars = []
        for par in self.required_params:
            pars.append(self.data_block.get(section_names.galaxy_bias,par))
        opts = {}
        for par in self.optional_params:
            opts[par] = self.data_block.get(section_names.galaxy_bias,par,self.optional_params[par])
        fsig = self.data_block.get(section_names.galaxy_rsd,'fsig',self.growth_rate*self.sigma8)
        f = fsig/self.sigma8
        self.theory.make_pltable(f,apar=1,aperp=1,**self.optional_kw)

        model_collection = ModelCollection()
        if self.with_correlation:
            self.model_correlation.eval = self.get_correlation_callable(pars,**opts)
            model_collection.set(self.model_correlation)
        if self.with_power:
            self.model_power.eval = self.get_power_callable(pars,**opts)
            model_collection.set(self.model_power)
        self.data_block[section_names.model,'collection'] = self.data_block.get(section_names.model,'collection',[]) + model_collection
