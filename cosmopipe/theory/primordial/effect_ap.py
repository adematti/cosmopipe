import numpy as np

from cosmoprimo import Cosmology, Background

from cosmopipe.lib.parameter import ParameterCollection
from cosmopipe import section_names


class EffectAP(object):

    def setup(self):
        self.zeff = self.data_block[section_names.survey_selection,'zeff']
        cosmo = self.data_block[section_names.fiducial_cosmology,'cosmo']
        self.engine = self.options.get('engine',None)
        ba = Background(cosmo,engine=self.engine,set_engine=False)
        self.hubble_rate = ba.efunc(self.zeff)
        self.comoving_angular_distance = ba.comoving_angular_distance(self.zeff)
        self.data_block[section_names.parameters,'derived'] = self.data_block.get(section_names.parameters,'derived',[])
        self.data_block[section_names.parameters,'derived'] += ParameterCollection([(section_names.effect_ap,name) for name in ['qpar','qperp']])

    def execute(self):
        cosmo = self.data_block[section_names.primordial_cosmology,'cosmo']
        ba = Background(cosmo,engine=self.engine,set_engine=False)
        self.data_block[section_names.effect_ap,'qpar'] = self.hubble_rate/ba.efunc(self.zeff)
        self.data_block[section_names.effect_ap,'qperp'] = ba.comoving_angular_distance(self.zeff)/self.comoving_angular_distance

    def cleanup(self):
        pass
