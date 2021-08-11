import numpy as np

from cosmopipe.lib.parameter import ParamName
from cosmopipe import section_names


class GalaxyRSD(object):

    def setup(self):
        self.radius_sigma = self.options.get('radius_sig',8.)
        self.data_block[section_names.galaxy_rsd,'radius_sig'] = self.radius_sigma
        derived = {ParamName(section_names.galaxy_rsd,'fsig')}
        self.data_block[section_names.parameters,'derived'] = self.data_block.get(section_names.parameters,'derived',set()) | derived

    def execute(self):
        cosmo = self.data_block[section_names.primordial_cosmology,'cosmo']
        zeff = self.data_block[section_names.survey_selection,'zeff']
        self.data_block[section_names.galaxy_rsd,'fsig'] = cosmo.get_fourier().sigma_rz(r=self.radius_sigma,z=zeff)

    def cleanup(self):
        pass
