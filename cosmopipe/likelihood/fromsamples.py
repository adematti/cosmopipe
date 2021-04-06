import logging

import numpy as np
from pypescript import BasePipeline
from pypescript.config import ConfigError

from cosmopipe import section_names
from cosmopipe.lib import utils
from cosmopipe.lib.samples import Samples
from cosmopipe.lib.parameter import ParamName

from .likelihood import GaussianLikelihood


class GaussianLikelihoodFromSamples(GaussianLikelihood):

    logger = logging.getLogger('GaussianLikelihoodFromSamples')

    def setup(self):
        self.samples_key = self.options.get('samples_key',None)
        self.samples_file = self.options.get_string('samples_file',None)
        if self.samples_file is None:
            if self.samples_key is None: self.samples_key = (section_names.likelihood,'samples')
            else:
                self.samples_key = utils.split_section_name(self.samples_key)
                if len(self.samples_key) == 1:
                    self.samples_key = (section_names.likelihood,) + self.samples_key
        elif self.samples_key is not None:
            raise ConfigError('Cannot provide both samples_key and samples_file.')
        BasePipeline.setup(self)
        self.parameters = self.options.get_list('parameters',None)
        self.set_samples()
        self.set_data()
        self.set_covariance()

    def set_samples(self):
        if self.samples_file:
            #samples = Samples.load_auto(self.samples_file,mpistate='broadcast')
            #samples = Samples.load_auto(self.samples_file,mpistate='scattered',mpicomm=self.mpicomm)
            samples = Samples.load_auto(self.samples_file,mpistate='broadcast',mpicomm=self.mpicomm)
        else:
            samples = self.pipe_block[self.samples_key]
        if self.parameters is None:
            self.parameters = samples.parameters.select(fixed=False)
        else:
            self.parameters = [samples.parameters[par] for par in self.parameters]
        self.parameter_names = [param.name.tuple for param in self.parameters]
        self.pipe_block[section_names.covariance,'invcov'] = utils.inv(samples.cov(self.parameters))
        self.nobs = self.pipe_block.get(section_names.covariance,'nobs',samples.neff())
        self.pipe_block[section_names.data,'y'] = samples.mean(self.parameters)
        #print(self.pipe_block[section_names.data,'y'])
        #self.pipe_block[section_names.data,'y'] = np.array([1.,1.])
        #self.pipe_block[section_names.data,'y'] = np.array([1.,1.,0.42538])

    def set_model(self):
        self.model = np.array([self.pipe_block[key] for key in self.parameter_names])
