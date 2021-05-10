import numpy as np

from cosmopipe import section_names


class GalaxyRSD(object):

    def setup(self):
        self.sigma_radius = self.options.get('sigma_radius',8.)

    def execute(self):
        cosmo = self.data_block[section_names.primordial_cosmology,'cosmology']
        zeff = self.data_block[section_names.survey_geometry,'zeff']
        self.data_block[section_names.galaxy_rsd,'fsig'] = cosmo.get_fourier().sigma_rz(r=self.sigma_radius,z=zeff)

    def cleanup(self):
        pass
