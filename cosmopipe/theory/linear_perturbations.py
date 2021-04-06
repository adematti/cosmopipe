import numpy as np
from pyccl import ccllib
import pyccl

from cosmopipe.lib.theory import PkLinear
from cosmopipe import section_names

from cosmopipe.lib.theory import PkEHNoWiggle


class LinearPerturbations(object):

    def setup(self):
        self.transfer_function = self.options.get('transfer_function','boltzmann_class')
        self.required_params = ['Omega_c','Omega_b', 'h', 'n_s', 'sigma8'] # A_s or sigma8
        self.optional_params = dict(Omega_k=0., Omega_g=None, Neff=3.046, m_nu=0., m_nu_type=None, w0=-1., wa=0., T_CMB=None)
        #self.pk_lin_callable = None

    def execute(self):
        kwargs = {}
        if 'omega_b' in self.data_block[section_names.cosmological_parameters]:
            h = self.data_block[section_names.cosmological_parameters,'h']
            self.data_block[section_names.cosmological_parameters,'Omega_b'] = self.data_block[section_names.cosmological_parameters,'omega_b']/h**2
        for par in self.required_params:
            kwargs[par] = self.data_block[section_names.cosmological_parameters,par]
        for par in self.optional_params:
            kwargs[par] = self.data_block.get(section_names.cosmological_parameters,par,self.optional_params[par])
        cosmo = pyccl.Cosmology(**kwargs,transfer_function=self.transfer_function)
        nk = ccllib.get_pk_spline_nk(cosmo.cosmo)
        kmax = cosmo.cosmo.spline_params.K_MAX_SPLINE
        kmin = 1e-5
        k = np.logspace(np.log10(kmin),np.log10(kmax),nk)/cosmo['h']
        a = self.data_block[section_names.background,'scale_factor']

        def _pk_lin_callable(k):
            return cosmo['h']**3*pyccl.linear_matter_power(cosmo,k*cosmo['h'],a)

        """
        if self.pk_lin_callable is None:
            self.pk_lin_callable = PkLinear(k=k,pk=_pk_lin_callable(k))

        self.data_block[section_names.linear_perturbations,'pk_callable'] = self.pk_lin_callable
        """
        pk_lin_callable = PkLinear(k=k,pk=_pk_lin_callable(k))

        self.data_block[section_names.linear_perturbations,'pk_callable'] = pk_lin_callable
        kwargs = {par:kwargs[par] for par in ['Omega_c','Omega_b','h']}
        self.data_block[section_names.linear_perturbations,'sound_horizon_drag'] = PkEHNoWiggle.sound_horizon(**kwargs)

    def cleanup(self):
        pass
