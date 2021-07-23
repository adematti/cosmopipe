import numpy as np

from cosmopipe.lib import theory
from cosmopipe.lib.theory.base import ProjectionBase, ModelCollection
from cosmopipe.lib.modules import ParameterizedModule

from cosmopipe import section_names


class AnisotropicScaling(ParameterizedModule):

    def setup(self):
        self.set_param_block()
        self.collection = ModelCollection()
        factor = 0.8
        for base,model in self.data_block[section_names.model,'collection']:
            base = base.copy()
            x = base.x
            base.x = x[(x > 1./factor*x[0]) & (x < factor*x[-1])]
            self.collection.set(theory.AnisotropicScaling(base=base))

    def execute(self):
        qpar = self.data_block[section_names.effect_ap,'qpar']
        qperp = self.data_block[section_names.effect_ap,'qperp']
        for base,model in self.collection:
            model.set_scaling(qpar=qpar,qperp=qperp)
        input_collection = self.data_block[section_names.model,'collection']
        for base,model in input_collection:
            if base in self.collection:
                self.collection[base].input_model = model
        self.data_block[section_names.model,'collection'] = input_collection + self.collection

    def cleanup(self):
        pass
