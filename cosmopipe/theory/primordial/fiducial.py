import numpy as np

from cosmoprimo import Cosmology

from cosmopipe import section_names


class Fiducial(object):

    def setup(self):
        params = {}
        params['engine'] = self.options.get('engine','class')
        for name,value in Cosmology.get_default_parameters(of='cosmology',include_conflicts=True).items():
            if name in self.options:
                params[name] = self.options[name]
        self.data_block[section_names.fiducial_cosmology,'cosmo'] = Cosmology(**params)

    def execute(self):
        pass

    def cleanup(self):
        pass
