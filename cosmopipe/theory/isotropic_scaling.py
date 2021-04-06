import numpy as np

from cosmopipe.lib import theory
from cosmopipe.lib.theory import PkLinear
from cosmopipe import section_names


class IsotropicScaling(object):

    def setup(self):
        self.scaling = theory.IsotropicScaling(pk=self.data_block[section_names.linear_perturbations,'pk_callable'],pivot=self.options.get_float('pivot',1./3.))
        k = self.scaling.input_pk.k
        factor = 0.8
        self.k = k[(k > 1./factor*k[0]) & (k < factor*k[-1])]
        self.pk_lin_callable = PkLinear.from_callable(self.k,self.scaling.pk)

    def execute(self):
        qpar = self.data_block[section_names.effect_ap,'qpar']
        qperp = self.data_block[section_names.effect_ap,'qperp']
        self.scaling.set_scaling(qpar=qpar,qperp=qperp)
        nqpar,nqperp = self.scaling.anisotropic_scaling()
        self.data_block[section_names.effect_ap,'qpar'] = nqpar
        self.data_block[section_names.effect_ap,'qperp'] = nqperp
        self.data_block[section_names.linear_perturbations,'pk_callable'] = self.pk_lin_callable
        self.data_block[section_names.linear_perturbations,'sigma8'] = self.pk_lin_callable.sigma8()

    def cleanup(self):
        pass
