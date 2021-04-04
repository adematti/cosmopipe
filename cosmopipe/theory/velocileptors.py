import numpy as np
from scipy import interpolate

from cosmopipe.lib.theory import PkLinear, PkEHNoWiggle
from cosmopipe import section_names


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
        options['threads'] = self.options.get_int('threads',1)
        return options

    def set_pklin(self):
        self.growth_rate = self.data_block[section_names.background,'growth_rate']
        pklin = self.data_block[section_names.linear_perturbations,'pk_callable']
        self.klin,self.pklin = pklin['k'],pklin['pk']

    def set_pknow(self):
        pknow = PkEHNoWiggle(k=self.klin)
        kwargs = {par: self.data_block[section_names.cosmological_parameters,par] for par in ['Omega_c','Omega_b', 'h', 'n_s', 'sigma8']}
        pknow.run(**kwargs)
        pknow.adjust_to_pk(self.data_block[section_names.linear_perturbations,'pk_callable'])
        self.pknow = pknow['pk']

    def execute(self):
        pars = []
        for par in self.required_params:
            pars.append(self.data_block.get(section_names.galaxy_bias,par))
        opts = {}
        for par in self.optional_params:
            opts[par] = self.data_block.get(section_names.galaxy_bias,par,self.optional_params[par])
        f = self.data_block.get(section_names.galaxy_rsd,'f',self.growth_rate)
        qpar = self.data_block.get(section_names.effect_ap,'qpar',1.)
        qperp = self.data_block.get(section_names.effect_ap,'qperp',1.)

        def pk_mu_callable(k,mu):
            # TODOs: in velocileptors, ask for kobs to avoid double interpolation
            # TODO: can be made much faster
            pkmu = np.array([self.theory.compute_redshift_space_power_at_mu(pars,f,mu_,apar=qpar,aperp=qperp,**opts,**self.optional_kw)[-1] for mu_ in mu]).T
            return interpolate.interp1d(self.theory.kv,pkmu,kind='cubic',axis=0,copy=False,bounds_error=True,assume_sorted=True)(k)

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
