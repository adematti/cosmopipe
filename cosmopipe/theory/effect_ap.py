import numpy as np
from pyccl import ccllib
import pyccl

from cosmopipe import section_names


class EffectAP(object):

    def setup(self):
        self.Omega_m = self.options['Omega_m']
        Omega_b = 0.05
        kwargs = dict(Omega_c=self.Omega_m-Omega_b,Omega_b=Omega_b,h=0.7,n_s=1,sigma8=1)
        self.cosmo = pyccl.Cosmology(**kwargs)

    def execute(self):
        self.a = self.data_block[section_names.background,'scale_factor']
        H = pyccl.h_over_h0(self.cosmo,self.a)
        self.data_block[section_names.effect_ap,'qpar'] = H/self.data_block[section_names.background,'hubble_rate']
        DA = pyccl.angular_diameter_distance(self.cosmo,self.a)*self.cosmo['h']
        self.data_block[section_names.effect_ap,'qperp'] = self.data_block[section_names.background,'angular_diameter_distance']/DA

    def cleanup(self):
        pass
