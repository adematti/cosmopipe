import numpy as np
from scipy import interpolate

from cosmoprimo import Cosmology, PowerSpectrumBAOFilter

from cosmopipe.lib.theory.base import BaseModel, ProjectionBasis, ModelCollection
from cosmopipe.lib.parameter import ParamName
from cosmopipe import section_names

from .ptmodule import PTModule


class Velocileptors(PTModule):

    def set_theory_options(self):
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
        self.theory_options = options

    def set_model(self, space=ProjectionBasis.POWER, mode=ProjectionBasis.MUWEDGE, projs=None):
        include = [ParamName(section_names.galaxy_bias,name) for name in self.required_params if name not in self.default_required_params]
        include += [ParamName(section_names.galaxy_bias,name) for name in self.optional_params]
        self.set_parameters(include=include + [(section_names.galaxy_rsd,'fsig')])
        basis = ProjectionBasis(x=self.theory.kv if space == ProjectionBasis.POWER else self.theory.rint,space=space,mode=mode,projs=projs,wa_order=0,**self.model_attrs)
        self.model = BaseModel(basis=basis)
        model_collection = self.data_block.get(section_names.model,'collection',[])
        model_collection += ModelCollection([self.model])
        self.data_block[section_names.model,'collection'] = model_collection

    def set_primordial(self):
        toret = super(Velocileptors,self).set_primordial()
        if toret:
            self.pknow = PowerSpectrumBAOFilter(self.pklin,engine='wallish2018').smooth_pk_interpolator()(self.klin)
            self.pklin = self.pklin(self.klin)
        return toret

    def set_model_callable(self, pars, f, **opts):

        def _make_model_callable(pars, f, **kwargs):
            def model_callable(k, mu, grid=True):
                if grid:
                    tmp = np.array([self.theory.compute_redshift_space_power_at_mu(pars,f,mu_,apar=1.,aperp=1.,**kwargs)[-1] for mu_ in mu]).T
                    if self.options.get('Gausstest_kc',None):
                        kc = self.options.get('Gausstest_kc',None)
                        sigk = self.options.get('Gausstest_sigk',None)
                        toret = np.array([1.0*20000*np.exp(-0.5*((k-kc)/sigk)**2) for mu_ in mu]).T
                    else:
                        toret = interpolate.interp1d(self.theory.kv,tmp,kind='cubic',axis=0,copy=False,bounds_error=True,assume_sorted=True)(k)
                else:
                    toret = np.empty(k.shape,dtype=k.dtype)
                    for imu,mu_ in enumerate(mu):
                        tmp = self.theory.compute_redshift_space_power_at_mu(pars,f,mu_,apar=1.,aperp=1.,**kwargs)[-1]
                        toret[:,imu] = interpolate.interp1d(self.theory.kv,tmp,kind='cubic',axis=0,copy=False,bounds_error=True,assume_sorted=True)(k[:,imu])
                return toret + self.data_shotnoise

            return model_callable

        self.model.eval = _make_model_callable(pars,f,**opts,**self.optional_kw)

    def execute(self):
        if self.set_primordial():
            self.set_theory()
        pars = []
        for par in self.required_params:
            pars.append(self.data_block.get(section_names.galaxy_bias,par))
        for par,value in self.default_required_params.items():
            pars.append(value)
        opts = {}
        for par in self.optional_params:
            opts[par] = self.data_block.get(section_names.galaxy_bias,par,self.optional_params[par])
        fsig = self.data_block[section_names.galaxy_rsd,'fsig']

        f = fsig/self.sigma
        self.set_model_callable(pars,f,**opts)

        #self.data_block[section_names.model,'collection'] = self.data_block.get(section_names.model,'collection',[]) + ModelCollection([self.model])

    def cleanup(self):
        pass


class EPTMoments(Velocileptors):

    _default_options = dict(rbao=110,kmin=1e-2,kmax=0.5,nk=100,beyond_gauss=True,one_loop=True,shear=True,third_order=True,cutoff=20,jn=5,N=4000,nthreads=1,extrap_min=-4,extrap_max=3,import_wisdom=False)

    def setup(self):
        reduced = self.options.get('reduced',True)
        self.set_theory_options()
        if self.theory_options['beyond_gauss']:
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
        self.optional_kw = dict(beyond_gauss=self.theory_options['beyond_gauss'],reduced=reduced)
        self.set_primordial()
        self.set_theory()
        self.set_model()

    def set_theory(self):
        from velocileptors.EPT.moment_expansion_fftw import MomentExpansion
        self.theory = MomentExpansion(self.klin,self.pklin,pnw=self.pknow,**self.theory_options)


class EPTFull(Velocileptors):

    #_default_options = dict(rbao=110,kmin=1e-2,kmax=0.5,nk=100,sbao=None,beyond_gauss=True,one_loop=True,shear=True,third_order=True,cutoff=20,jn=5,N=4000,nthreads=1,extrap_min=-4,extrap_max=3,import_wisdom=False)
    _default_options = dict(rbao=110,kmin=1e-2,kmax=0.5,nk=100,sbao=None,beyond_gauss=True,one_loop=True,shear=True,cutoff=20,jn=5,N=4000,nthreads=1,extrap_min=-4,extrap_max=3,import_wisdom=False)

    def setup(self):
        reduced = self.options.get('reduced',True)
        self.set_theory_options()
        self.required_params = ['b1', 'b2', 'bs', 'b3', 'alpha0', 'alpha2', 'alpha4', 'alpha6', 'sn', 'sn2', 'sn4']
        self.default_required_params = {}
        self.optional_params = dict(bFoG=0.)
        self.optional_kw = dict()
        self.set_primordial()
        self.set_theory()
        self.set_model()

    def set_theory(self):
        from velocileptors.EPT.ept_fullresum_fftw import REPT
        self.theory = REPT(self.klin,self.pklin,pnw=self.pknow,**self.theory_options)


class LPTMoments(Velocileptors):

    _default_options = dict(kmin=5e-3,kmax=0.3,nk=50,beyond_gauss=False,one_loop=True,shear=True,third_order=True,cutoff=10,jn=5,N=2000,nthreads=None,extrap_min=-5,extrap_max=3,import_wisdom=False)

    def setup(self):
        reduced = self.options.get_bool('reduced',True)
        self.set_theory_options()

        if self.theory_options['beyond_gauss']:
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
        if not self.theory_options['third_order']:
            self.default_required_params['b3'] = 0.
            if not self.theory_options['shear']:
                self.default_required_params['bs'] = 0.
        self.optional_params = dict(counterterm_c3=0.)
        self.optional_kw = dict(ngauss=4,reduced=reduced)
        self.set_primordial()
        self.set_theory()
        self.set_model()

    def set_theory(self):
        from velocileptors.LPT.moment_expansion_fftw import MomentExpansion
        self.theory = MomentExpansion(self.klin,self.pklin,**self.theory_options)


class LPTFourierStreaming(Velocileptors):

    _default_options = dict(kmin=1e-3,kmax=3,nk=100,beyond_gauss=False,one_loop=True,shear=True,third_order=True,cutoff=10,jn=5,N=2000,nthreads=None,extrap_min=-5,extrap_max=3,import_wisdom=False)

    def setup(self):
        # jn = 5
        self.set_theory_options()
        # b3 if in third order, bs if shear bias
        self.required_params = ['b1', 'b2', 'bs', 'b3','alpha', 'alpha_v', 'alpha_s0', 'alpha_s2', 'sn', 'sv', 'sigma0_stoch']
        self.default_required_params = {}
        if not self.theory_options['third_order']:
            self.default_required_params['b3'] = 0.
            if not self.theory_options['shear']:
                self.default_required_params['bs'] = 0.
        self.optional_params = dict(counterterm_c3=0)
        self.optional_kw = dict()
        self.set_primordial()
        self.set_theory()
        self.set_model()

    def set_theory(self):
        from velocileptors.LPT.fourier_streaming_model_fftw import FourierStreamingModel
        self.theory = FourierStreamingModel(self.klin,self.pklin,**self.theory_options)


class LPTGaussianStreaming(Velocileptors):

    #_default_options = dict(kmin=1e-3,kmax=3,nk=200,jn=10,cutoff=20,beyond_gauss=False,one_loop=True,shear=True,third_order=True,N=2000,nthreads=None,extrap_min=-5,extrap_max=3,import_wisdom=False)
    _default_options = dict(kmin=1e-3,kmax=3,nk=200,jn=10,cutoff=20,beyond_gauss=False,one_loop=True,shear=True,N=2000,nthreads=None,extrap_min=-5,extrap_max=3,import_wisdom=False)

    def setup(self):
        # kmin = 3e-3, kmax=0.5, nk = 100, kswitch=1e-2, jn = 5, cutoff=20
        self.set_theory_options()
        # alpha_s0 and alpha_s2 to be zeros
        self.required_params = ['b1', 'b2', 'bs', 'b3', 'alpha', 'alpha_v', 'alpha_s0', 'alpha_s2', 's2FoG']
        self.default_required_params = {}
        if not self.theory_options.get('third_order',True):
            self.default_required_params['b3'] = 0.
            if not self.theory_options['shear']:
                self.default_required_params['bs'] = 0.
        self.optional_params = dict()
        self.optional_kw = dict(rwidth=100,Nint=10000,ngauss=4,update_cumulants=False)
        self.set_primordial()
        self.set_theory()
        self.set_model(space=ProjectionBasis.CORRELATION,mode=ProjectionBasis.MULTIPOLE,projs=(0,2,4))

    def set_theory(self):
        from velocileptors.LPT.gaussian_streaming_model_fftw import GaussianStreamingModel
        self.theory = GaussianStreamingModel(self.klin,self.pklin,**self.theory_options)

    def set_model_callable(self, pars, f, **opts):

        def _make_model_callable(pars, f, **kwargs):

            def model_callable(s):
                xi = []
                for ss in s:
                    xi0,xi2,xi4 = self.theory.compute_xi_ell(ss,f,*pars,apar=1.,aperp=1.,**kwargs)
                    xi.append([xi0,xi2,xi4])
                return np.array(xi)

            return model_callable

        self.model.eval = _make_model_callable(pars,f,**opts,**self.optional_kw)


class LPTDirect(Velocileptors):

    _default_options = dict(third_order=True,shear=True,one_loop=True,kIR=None,cutoff=10,jn=5,N=2000,nthreads=None,extrap_min=-5,extrap_max=3)

    def setup(self):
        # kmin = 3e-3, kmax=0.5, nk = 100, kswitch=1e-2, jn = 5, cutoff=20
        self.set_theory_options()
        output = self.options.get_list('output',['power'])
        self.with_power = 'power' in output
        self.with_correlation = 'correlation' in output
        # alpha_s0 and alpha_s2 to be zeros
        self.required_params = ['b1', 'b2', 'bs', 'b3', 'alpha0', 'alpha2', 'alpha4', 'alpha6', 'sn', 'sn2', 'sn4']
        self.default_required_params = {}
        if not self.theory_options['third_order']:
            self.default_required_params['b3'] = 0.
            if not self.theory_options['shear']:
                self.default_required_params['bs'] = 0.
        self.optional_params = dict()
        self.optional_kw = dict(ngauss=3,kv=None,kmin=5e-3 if self.with_correlation else 1e-2,kmax=1.0 if self.with_correlation else 0.25,nk=60 if self.with_correlation else 50,nmax=4)
        self.set_primordial()
        self.set_theory()
        if self.with_correlation:
            self.set_model(space=ProjectionBasis.CORRELATION,mode=ProjectionBasis.MULTIPOLE,projs=(0,2,4),**self.model_attrs)
            self.model_correlation = self.model
        if self.with_power:
            self.set_model(space=ProjectionBasis.POWER,mode=ProjectionBasis.MULTIPOLE,projs=(0,2,4),**self.model_attrs)
            self.model_power = self.model

    def set_theory(self):
        from velocileptors.LPT.lpt_rsd_fftw import LPT_RSD
        self.theory = LPT_RSD(self.klin,self.pklin,**self.theory_options)

    def set_model_callable(self, pars, f, **opts):

        self.theory.make_pltable(f,apar=1,aperp=1,**self.optional_kw)

        if self.with_correlation:

            def _make_model_callable(pars, f, **kwargs):

                def model_callable(s):
                    xiells = lpt.combine_bias_terms_xiell(pars,**opts)
                    return np.array([interpolate.interp1d(xiell[0],xiell[1],kind='cubic',axis=0,copy=False,bounds_error=True,assume_sorted=True)(s) for xiell in xiells])

                return model_callable

            self.correlation_model.eval = _make_model_callable(pars,f,**opts)

        if self.with_power:

            def _make_model_callable(pars, f, **kwargs):

                def model_callable(k):
                    kl, p0, p2, p4 = self.theory.combine_bias_terms_pkell(pars,**opts)
                    p0 += self.data_shotnoise
                    pkells = np.array([p0,p2,p4]).T
                    return interpolate.interp1d(kl,pkells,kind='cubic',axis=0,copy=False,bounds_error=True,assume_sorted=True)(k)

                return model_callable

            self.power_model.eval = _make_model_callable(pars,f,**opts)
