import logging

import numpy as np
from pypescript import BasePipeline, ConfigError, syntax

from cosmopipe import section_names
from cosmopipe.lib import utils
from cosmopipe.lib.samples import Samples
from cosmopipe.lib.parameter import ParamName

from .likelihood import GaussianLikelihood


class GaussianLikelihoodFromSamples(GaussianLikelihood):

    logger = logging.getLogger('GaussianLikelihoodFromSamples')

    def setup(self):
        BasePipeline.setup(self)
        self.samples_load = self.options.get('samples_load','samples')
        self.parameters = self.options.get_list('parameters',None)
        self.set_samples()
        self.set_data()
        self.set_covariance()

    def set_samples(self):
        samples = syntax.load_auto(self.samples_load,data_block=self.pip_block,default_section=section_names.likelihood,
                                    loader=Samples.load_auto,squeeze=True,mpistate='broadcast',mpicomm=self.mpicomm)
        if self.parameters is None:
            self.parameters = samples.parameters.select(fixed=False)
        else:
            self.parameters = [samples.parameters[par] for par in self.parameters]
        self.parameter_names = [param.name.tuple for param in self.parameters]
        self.pipe_block[section_names.covariance,'invcov'] = utils.inv(samples.cov(self.parameters))
        self.nobs = self.pipe_block.get(section_names.covariance,'nobs',samples.sum('metrics.fweight'))
        self.pipe_block[section_names.data,'y'] = samples.mean(self.parameters)
        #print(self.pipe_block[section_names.data,'y'])
        #self.pipe_block[section_names.data,'y'] = np.array([1.,1.])
        #self.pipe_block[section_names.data,'y'] = np.array([1.,1.,0.42538])

    def set_model(self):
        self.model = np.array([self.pipe_block[key] for key in self.parameter_names])
