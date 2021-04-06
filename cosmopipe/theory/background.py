import numpy as np
from pyccl import ccllib
import pyccl

from cosmopipe.lib.theory import PkLinear, PkEHNoWiggle
from cosmopipe import section_names


class Background(object):

    def setup(self):
        self.required_params = ['Omega_c','Omega_b', 'h', 'n_s', 'sigma8'] # A_s or sigma8
        self.optional_params = dict(Omega_k=0., Omega_g=None, Neff=3.046, m_nu=0., m_nu_type=None, w0=-1., wa=0., T_CMB=None)

    def execute(self):
        kwargs = {}
        if 'omega_b' in self.data_block[section_names.cosmological_parameters]:
            h = self.data_block[section_names.cosmological_parameters,'h']
            self.data_block[section_names.cosmological_parameters,'Omega_b'] = self.data_block[section_names.cosmological_parameters,'omega_b']/h**2
        for par in self.required_params:
            kwargs[par] = self.data_block[section_names.cosmological_parameters,par]
        for par in self.optional_params:
            kwargs[par] = self.data_block.get(section_names.cosmological_parameters,par,self.optional_params[par])

        cosmo = pyccl.Cosmology(**kwargs)

        a = self.data_block[section_names.background,'scale_factor']
        self.data_block[section_names.background,'growth_rate'] = pyccl.background.growth_rate(cosmo,a)
        self.data_block[section_names.background,'hubble_rate'] = pyccl.background.h_over_h0(cosmo,a)
        #self.data_block[section_names.background,'comoving_radial_distance'] = pyccl.background.comoving_radial_distance(cosmo,a)*cosmo['h']
        self.data_block[section_names.background,'comoving_angular_distance'] = pyccl.background.comoving_angular_distance(cosmo,a)*cosmo['h']
        kwargs = {par:kwargs[par] for par in ['Omega_c','Omega_b','h']}
        self.data_block[section_names.linear_perturbations,'sound_horizon_drag'] = PkEHNoWiggle.sound_horizon(**kwargs)

    def cleanup(self):
        pass
