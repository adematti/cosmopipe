import numpy as np

from cosmopipe.lib import theory
from cosmopipe.lib.primordial import PowerSpectrumInterpolator2D
from cosmopipe import section_names


class IsotropicScaling(object):

    def setup(self):
        self.scaling = theory.IsotropicScaling(pk=self.data_block[section_names.primordial_perturbations,'pk_callable'],
                                                pivot=self.options.get_float('pivot',1./3.))
        k = self.scaling.input_pk.k
        factor = 0.8
        self.k = k[(k > 1./factor*k[0]) & (k < factor*k[-1])]
        self.pk_lin_callable = PowerSpectrumInterpolator2D.from_callable(k=self.k,pk_callable=self.scaling.pk)

    def execute(self):
        qpar = self.data_block[section_names.effect_ap,'qpar']
        qperp = self.data_block[section_names.effect_ap,'qperp']
        self.scaling.set_scaling(qpar=qpar,qperp=qperp)
        nqpar,nqperp = self.scaling.anisotropic_scaling()
        self.data_block[section_names.effect_ap,'qpar'] = nqpar
        self.data_block[section_names.effect_ap,'qperp'] = nqperp
        zeff = self.data_block[section_names.survey_geometry,'zeff']
        self.data_block[section_names.primordial_perturbations,'pk_callable'] = self.pk_lin_callable

    def cleanup(self):
        pass
