import numpy as np

from cosmopipe.lib import theory
from cosmopipe.lib.theory.base import ProjectionBasis, ModelCollection
from cosmopipe.parameters import ParameterizedModule

from cosmopipe import section_names


class AnisotropicScaling(ParameterizedModule):

    def setup(self):
        self.set_parameters()
        factor = 0.8
        self.model_collection = self.data_block[section_names.model,'collection']
        scaling_collection = ModelCollection()
        for basis,model in self.model_collection.items():
            basis = basis.copy()
            x = basis.x
            basis.x = x[(x > 1./factor*x[0]) & (x < factor*x[-1])]
            scaling_collection.set(theory.AnisotropicScaling(model=model,basis=basis)) # this still references the "no ap" model
        self.model_collection.clear()
        self.model_collection += scaling_collection

    def execute(self):
        qpar = self.data_block[section_names.effect_ap,'qpar']
        qperp = self.data_block[section_names.effect_ap,'qperp']
        for model in self.model_collection:
            model.set_scaling(qpar=qpar,qperp=qperp)

    def cleanup(self):
        pass
