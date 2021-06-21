import numpy as np
from scipy import interpolate

from cosmopipe import section_names
from cosmopipe.lib.primordial import Cosmology, PowerSpectrumBAOFilter
from cosmopipe.lib.theory.projection import ProjectionBase


class Velocileptors(object):

    def get_theory_options(self):
        options = {}
        options['one_loop'] = self.options.get_bool('one_loop',True)
        options['third_order'] = self.options.get_bool('third_order',True)
        options['beyond_gauss'] = self.options.get_bool('beyond_gauss',True)
        options['kmin'] = self.options.get_float('kmin',5e-3)
        options['kmax'] = self.options.get_float('kmax',0.25)
        options['nk'] = self.options.get_int('nk',120)
        options['cutoff'] = self.options.get_int('cutoff',100)
        options['N'] = self.options.get_int('N',2000)
        options['shear'] = self.options.get_bool('shear',True)
        options['extrap_min'] = -6
        options['extrap_max'] = 2
        options['threads'] = self.options.get_int('nthreads',1)
        self.data_block[section_names.model,'y_base'] = ProjectionBase('muwedge')
        return options

    def set_pklin(self):
        self.zeff = self.data_block[section_names.survey_geometry,'zeff']
        pklin = self.data_block[section_names.primordial_perturbations,'pk_callable']
        self.sigma8 = pklin.sigma8_z(self.zeff)
        self.klin,self.pklin = pklin.k,pklin(pklin.k,z=self.zeff)
        fo = self.data_block[section_names.primordial_cosmology,'cosmo'].get_fourier()
        self.growth_rate = fo.sigma8_z(self.zeff,of='theta_cb')/fo.sigma8_z(self.zeff,of='delta_cb')

    def set_pknow(self):
        pklin = self.data_block[section_names.primordial_perturbations,'pk_callable'].to_1d(z=self.zeff)
        self.pknow = PowerSpectrumBAOFilter(pklin,engine='wallish2018').smooth_pk_interpolator()(self.klin)

    def execute(self):
        pars = []
        for par in self.required_params:
            pars.append(self.data_block.get(section_names.galaxy_bias,par))
        opts = {}
        for par in self.optional_params:
            opts[par] = self.data_block.get(section_names.galaxy_bias,par,self.optional_params[par])
        fsig = self.data_block.get(section_names.galaxy_rsd,'fsig',self.growth_rate*self.sigma8)
        f = fsig/self.sigma8
        qpar = self.data_block.get(section_names.effect_ap,'qpar',1.)
        qperp = self.data_block.get(section_names.effect_ap,'qperp',1.)

        def pk_mu_callable(k,mu):
            # TODOs: in velocileptors, ask for kobs to avoid double interpolation
            # TODO: can be made much faster
            pkmu = np.array([self.theory.compute_redshift_space_power_at_mu(pars,f,mu_,apar=qpar,aperp=qperp,**opts,**self.optional_kw)[-1] for mu_ in mu]).T
            return interpolate.interp1d(self.theory.kv,pkmu,kind='cubic',axis=0,copy=False,bounds_error=True,assume_sorted=True)(k)

        self.data_block[section_names.model,'x'] = self.theory.kv
        self.data_block[section_names.model,'y_callable'] = self.data_block[section_names.galaxy_power,'pk_mu_callable'] = pk_mu_callable

    def cleanup(self):
        pass


class EPTMoments(Velocileptors):

    def setup(self):
        reduced = self.options.get_bool('reduced',True)
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
        self.optional_params = dict(counterterm_c3=0)
        self.optional_kw = dict(beyond_gauss=options['beyond_gauss'],reduced=reduced)


class EPTFull(Velocileptors):

    def setup(self):
        reduced = self.options.get_bool('reduced',True)
        options = self.get_theory_options()
        options.pop('third_order')
        self.set_pklin()
        self.set_pknow()
        from velocileptors.EPT.ept_fullresum_fftw import REPT
        self.theory = REPT(self.klin,self.pklin,pnw=self.pknow,**options)
        self.required_params = ['b1', 'b2', 'bs', 'b3', 'alpha0', 'alpha2', 'alpha4', 'alpha6', 'sn', 'sn2', 'sn4']
        self.optional_params = dict(bFoG=0)
        self.optional_kw = dict()


class LPTMoments(Velocileptors):

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
        self.optional_params = dict(counterterm_c3=0)
        self.optional_kw = dict(reduced=reduced)
