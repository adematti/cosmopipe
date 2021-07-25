import numpy as np

from cosmoprimo import PowerSpectrumInterpolator2D

from cosmopipe.lib import theory
from cosmopipe.lib.theory.base import ProjectionBase
from cosmopipe.lib.modules import ParameterizedModule

from cosmopipe import section_names


class IsotropicScaling(ParameterizedModule):

    def setup(self):
        self.set_param_block()
        input_model = self.data_block[section_names.primordial_perturbations,'pk_callable']
        self.scaling = theory.IsotropicScaling(model=input_model,
                                                base=ProjectionBase(space=ProjectionBase.POWER),
                                                pivot=self.options.get_float('pivot',1./3.))
        k = input_model.k
        factor = 0.8
        k = k[(k > 1./factor*k[0]) & (k < factor*k[-1])] # just to mimic original model
        self.pk_lin_callable = PowerSpectrumInterpolator2D.from_callable(k=k,pk_callable=self.scaling.eval)
        self.data_block[section_names.primordial_perturbations,'pk_callable'] = self.pk_lin_callable

    def execute(self):
        qpar = self.data_block[section_names.effect_ap,'qpar']
        qperp = self.data_block[section_names.effect_ap,'qperp']
        self.scaling.set_scaling(qpar=qpar,qperp=qperp)
        nqpar,nqperp = self.scaling.anisotropic_scaling()
        self.data_block[section_names.effect_ap,'qpar'] = nqpar
        self.data_block[section_names.effect_ap,'qperp'] = nqperp

    def cleanup(self):
        pass
