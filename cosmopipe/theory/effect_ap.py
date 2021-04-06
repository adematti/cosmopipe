import numpy as np
from pyccl import ccllib
import pyccl

from cosmopipe import section_names


class EffectAP(object):

    def setup(self):
        Omega_m = self.options['Omega_m']
        Omega_b = 0.05
        kwargs = dict(Omega_c=Omega_m-Omega_b,Omega_b=Omega_b,h=0.7,n_s=1,sigma8=1)
        cosmo = pyccl.Cosmology(**kwargs)
        a = self.data_block[section_names.background,'scale_factor']
        self.hubble_rate = pyccl.background.h_over_h0(cosmo,a)
        self.comoving_angular_distance = pyccl.background.comoving_angular_distance(cosmo,a)*cosmo['h']

    def execute(self):
        hubble_rate = self.data_block[section_names.background,'hubble_rate']
        comoving_angular_distance = self.data_block[section_names.background,'comoving_angular_distance']
        self.data_block[section_names.effect_ap,'qpar'] = self.hubble_rate/hubble_rate
        self.data_block[section_names.effect_ap,'qperp'] = comoving_angular_distance/self.comoving_angular_distance

    def cleanup(self):
        pass


"""
class EffectAP(object):

    def setup(self):
        Omega_m = self.options['Omega_m']
        Omega_b = 0.05
        kwargs = dict(Omega_c=Omega_m-Omega_b,Omega_b=Omega_b,h=0.7,n_s=1,sigma8=1)
        self.cosmo = pyccl.Cosmology(**kwargs)

    def execute(self):
        a = self.data_block[section_names.background,'scale_factor']
        H = pyccl.background.h_over_h0(self.cosmo,a)
        self.data_block[section_names.effect_ap,'qpar'] = H/self.data_block[section_names.background,'hubble_rate']
        DM = pyccl.background.comoving_angular_distance(self.cosmo,a)*self.cosmo['h']
        self.data_block[section_names.effect_ap,'qperp'] = self.data_block[section_names.background,'comoving_angular_distance']/DM

    def cleanup(self):
        pass
"""
