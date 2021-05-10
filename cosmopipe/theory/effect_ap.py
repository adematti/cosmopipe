import numpy as np

from cosmopipe.lib.primordial import Cosmology
from cosmopipe import section_names


class EffectAP(object):

    def setup(self):
        self.zeff = self.data_block[section_names.survey_geometry,'zeff']
        params = {}
        params['engine'] = self.options.get('engine','class')
        for name,value in Cosmology.get_default_parameters(of='cosmology',include_conflicts=True).items():
            if name in self.options:
                params[name] = self.options[name]
        cosmo = Cosmology(**params)
        ba = cosmo.get_background()
        self.hubble_rate = ba.efunc(self.zeff)
        self.comoving_angular_distance = ba.comoving_angular_distance(self.zeff)

    def execute(self):
        ba = self.data_block[section_names.primordial_cosmology,'cosmo'].get_background()
        self.data_block[section_names.effect_ap,'qpar'] = self.hubble_rate/ba.efunc(self.zeff)
        self.data_block[section_names.effect_ap,'qperp'] = ba.comoving_angular_distance(self.zeff)/self.comoving_angular_distance

    def cleanup(self):
        pass
