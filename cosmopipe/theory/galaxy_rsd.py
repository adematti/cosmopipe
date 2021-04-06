import numpy as np

from cosmopipe import section_names


class GalaxyRSD(object):

    def setup(self):
        pass

    def execute(self):
        f = self.data_block[section_names.background,'growth_rate']
        sig = self.data_block[section_names.linear_perturbations,'pk_callable'].sigma8()
        self.data_block[section_names.galaxy_rsd,'fsig'] = f*sig

    def cleanup(self):
        pass
