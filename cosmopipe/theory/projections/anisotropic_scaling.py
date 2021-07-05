import numpy as np

from cosmoprimo import PowerSpectrumInterpolator2D

from cosmopipe.lib import theory
from cosmopipe.lib.theory.base import ProjectionBase
from cosmopipe import section_names


class AnisotropicScaling(object):

    def setup(self):
        self.collection = self.data_block[section_names.model,'collection'].copy()
        factor = 0.8
        for base,model in self.collection:
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
            self.collection[base].input_model = model
        self.data_block[section_names.model,'collection'] = self.collection

    def cleanup(self):
        pass
