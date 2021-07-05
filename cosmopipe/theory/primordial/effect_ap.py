import numpy as np

from cosmopipe.lib.primordial import Cosmology
from cosmopipe import section_names


class EffectAP(object):

    def setup(self):
        self.zeff = self.data_block[section_names.survey_geometry,'zeff']
        cosmo = self.data_block[section_names.fiducial_cosmology,'cosmo']
        self.hubble_rate = ba.efunc(self.zeff)
        self.comoving_angular_distance = ba.comoving_angular_distance(self.zeff)

    def execute(self):
        ba = self.data_block[section_names.primordial_cosmology,'cosmo'].get_background()
        self.data_block[section_names.effect_ap,'qpar'] = self.hubble_rate/ba.efunc(self.zeff)
        self.data_block[section_names.effect_ap,'qperp'] = ba.comoving_angular_distance(self.zeff)/self.comoving_angular_distance

    def cleanup(self):
        pass
