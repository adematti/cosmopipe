import functools

import numpy as np
from scipy import interpolate

from cosmopipe.lib.utils import BaseClass
from cosmopipe.lib.primordial import PowerSpectrumInterpolator1D
from .fog import get_FoG


class BasePTModel(BaseClass):

    def __init__(self, pklin, klin=None, FoG='gaussian'):
        if klin is None:
            if callable(pklin):
                self.pk_linear = pklin
            else:
                raise ValueError('Input pklin should be a PowerSpectrumInterpolator1D instance if no k provided.')
        else:
            self.pk_linear = PowerSpectrumInterpolator1D(k=klin,pk=pklin)
        self.FoG = FoG
        if FoG is not None:
            self.FoG = get_FoG(FoG)
