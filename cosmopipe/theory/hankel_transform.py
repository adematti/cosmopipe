import numpy as np
from scipy import interpolate

from cosmoprimo import CorrelationToPower, PowerToCorrelation

from cosmopipe.lib import utils
from cosmopipe.lib.theory.projection import DataVectorProjection, ProjectionBase
from cosmopipe.lib.utils import BaseClass


class HankelTransform(object):

    def setup(self):
        integration = self.options.get('integration',None)
        nx = self.options.get('nx',1024)
        self.model_base = self.data_block[section_names.model,'y_base']
        x = self.data_block[section_names.model,'x']
        xmin,xmax = x.min(),x.max()
        self.x = np.logspace(np.log10(xmin),np.log10(xmax),nx)
        if self.model_base.mode == ProjectionBase.MUWEDGE:
            self.ells = self.options.get('ells',(0,2,4))
            projs = ['ell_{:d}'.format(ell) for ell in self.ells]
            self.projection = DataVectorProjection(x=x,projs=projs,model_base=self.model_base,integration=integration)
        else:
            self.ells = self.model_base.projs
        q = self.options.get('q',1.5)
        if self.options.get('pk2xi',True):
            self.fftlog = PowerToCorrelation(self.x,ell=self.ells,q=q)
        else:
            self.fftlog = CorrelationToPower(self.x,ell=self.ells,q=q)
        self.data_block[section_names.model,'y_base'] = ProjectionBase('multipole',self.ells)

    def execute(self):
        fun = self.data_block[section_names.model,'y_callable']
        if self.model_base.mode == ProjectionBase.MUWEDGE:
            fun = self.projection(fun,concatenate=False)
        else:
            fun = fun(x)
        x, fun = self.fftlog(fun)
        self.data_block[section_names.model,'x'] = x[0] # as many x's as ells
        self.data_block[section_names.model,'y_callable'] = interpolate.interp1d(x,fun.T,axis=0,kind='cubic',bounds_error=True,assume_sorted=True)

    def cleanup(self):
        pass
