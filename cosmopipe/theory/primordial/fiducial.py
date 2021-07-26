import numpy as np

from cosmoprimo import Cosmology

from cosmopipe.lib.modules import ParameterizedModule
from cosmopipe import section_names


class Fiducial(ParameterizedModule):

    def setup(self):
        self.set_parameters()
        self.calculation_params = {}
        self.calculation_params['engine'] = self.options.get('engine','class')
        for name,value in Cosmology.get_default_parameters(of='calculation',include_conflicts=True).items():
            self.calculation_params[name] = self.options.get(name,value)
        self.optional_params = Cosmology.get_default_parameters(of='cosmology')

        params = {}
        for par in self.optional_params:
            try:
                params[par] = self.data_block.get(section_names.fiducial_cosmology,par)
            except KeyError:
                pass
        cosmo = Cosmology(**params,**self.calculation_params)

        self.data_block[section_names.fiducial_cosmology,'cosmo'] = cosmo

    def execute(self):
        pass

    def cleanup(self):
        pass
