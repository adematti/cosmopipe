import numpy as np

from cosmoprimo import Cosmology

from cosmopipe.lib.modules import ParameterizedModule
from cosmopipe import section_names


class Primordial(ParameterizedModule):

    def setup(self):
        self.set_parameters()
        self.compute = self.options.get('compute',None)
        self.calculation_params = {}
        self.calculation_params['engine'] = self.options.get('engine','class')
        for name,value in Cosmology.get_default_parameters(of='calculation',include_conflicts=True).items():
            self.calculation_params[name] = self.options.get(name,value)
        self.optional_params = Cosmology.get_default_parameters(of='cosmology')

    def execute(self):
        params = {}
        for par in self.optional_params:
            try:
                params[par] = self.data_block.get(section_names.primordial_cosmology,par)
            except KeyError:
                pass
        cosmo = Cosmology(**params,**self.calculation_params)
        self.data_block[section_names.primordial_cosmology,'cosmo'] = cosmo
        fo = cosmo.get_fourier()
        if self.compute == 'pk_m':
            self.data_block[section_names.primordial_perturbations,'pk_callable'] = fo.pk_interpolator(of='delta_m')
        elif self.compute == 'pk_cb':
            self.data_block[section_names.primordial_perturbations,'pk_callable'] = fo.pk_interpolator(of='delta_cb')

    def cleanup(self):
        pass
