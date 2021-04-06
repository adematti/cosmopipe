import numpy as np
from pyccl import ccllib
import pyccl

from cosmopipe.lib.theory import PkEHNoWiggle
from cosmopipe import section_names


class GalaxyBAO(object):

    def setup(self):
        kwargs = dict(Omega_c=self.options['Omega_c'],Omega_b=self.options['Omega_b'],h=self.options['h'])
        cosmo = pyccl.Cosmology(**kwargs,n_s=1,sigma8=1)
        self.rdrag = PkEHNoWiggle.sound_horizon(**kwargs)
        self.rdrag = 144.16545636718809*self.options['h'] # hack because pyccl does not provide it
        a = self.data_block[section_names.background,'scale_factor']
        self.hubble_rate = pyccl.background.h_over_h0(cosmo,a)
        self.comoving_angular_distance = pyccl.background.comoving_angular_distance(cosmo,a)*cosmo['h']

    def execute(self):
        hubble_rate = self.data_block[section_names.background,'hubble_rate']
        comoving_angular_distance = self.data_block[section_names.background,'comoving_angular_distance']
        rdrag = self.data_block[section_names.linear_perturbations,'sound_horizon_drag']
        self.data_block[section_names.effect_ap,'qpar'] = self.hubble_rate*self.rdrag/hubble_rate/rdrag
        self.data_block[section_names.effect_ap,'qperp'] = comoving_angular_distance*self.rdrag/self.comoving_angular_distance/rdrag

    def cleanup(self):
        pass
