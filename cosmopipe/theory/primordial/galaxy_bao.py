import numpy as np

from cosmopipe.lib.primordial import Cosmology
from cosmopipe import section_names


class GalaxyBAO(object):

    def setup(self):
        self.zeff = self.data_block[section_names.survey_geometry,'zeff']
        cosmo = self.data_block[section_names.fiducial_cosmology,'cosmo']
        th = cosmo.get_thermodynamics()
        ba = cosmo.get_background()
        self.hubble_rate = ba.efunc(self.zeff)
        self.comoving_transverse_distance = ba.comoving_angular_distance(self.zeff)
        self.rs_drag = th.rs_drag

    def execute(self):
        cosmo = self.data_block[section_names.primordial_cosmology,'cosmo']
        th = cosmo.get_thermodynamics()
        ba = cosmo.get_background()
        self.data_block[section_names.effect_ap,'qpar'] = self.hubble_rate*self.rs_drag/ba.efunc(self.zeff)/th.rs_drag
        self.data_block[section_names.effect_ap,'qperp'] = ba.comoving_angular_distance(self.zeff)*self.rs_drag/self.comoving_angular_distance/th.rs_drag

    def cleanup(self):
        pass
