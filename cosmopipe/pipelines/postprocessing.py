import logging

import numpy as np
from pypescript import MPIPipeline

from cosmopipe import section_names
from cosmopipe.lib import syntax
from cosmopipe.lib.samples import Samples, Profiles


class SamplesPostprocessing(MPIPipeline):

    logger = logging.getLogger('SamplesPostprocessing')

    def setup(self):
        self.burnin = self.options.get('burnin',None)
        self.mode = self.options.get('mode','mean')
        self.samples_load = self.options.get('samples_load','samples')
        chains = syntax.load_auto(self.samples_load,data_block=self.data_block,default_section=section_names.likelihood,loader=Samples.load_auto)
        if self.burnin is not None:
            chains = [samples.remove_burnin(self.burnin) for samples in chains]
        columns = chains[0].columns()
        data_block_iter = {}
        if self.mode == 'mean':
            for col in columns:
                data_block_iter[col.tuple] = [np.average([samples.average(col) for samples in chains],weights=[samples.sum('metrics.weight') for samples in chains])]
            self.options['iter'] = [0]
        self.options[syntax.datablock_iter] = data_block_iter
        super(SamplesPostprocessing,self).setup()


class ProfilesPostprocessing(MPIPipeline):

    logger = logging.getLogger('ProfilesPostprocessing')

    def setup(self):
        self.mode = self.options.get('mode','best')
        self.profiles_load = self.options.get('profiles_load','profiles')
        profiles = syntax.load_auto(self.profiles_load,data_block=self.data_block,default_section=section_names.likelihood,loader=Profiles.load_auto)
        parameters = profiles[0].parameters
        data_block_iter = {}
        if self.mode == 'best':
            argmin = np.argmin([profile.metrics['fval'][profile.argmin()] for profile in profiles])
            for param in parameters:
                data_block_iter[param.name.tuple] = [profiles[argmin].bestfit[param][profiles[argmin].argmin()]]
            self.options['iter'] = [0]
        self.options[syntax.datablock_iter] = data_block_iter
        super(ProfilesPostprocessing,self).setup()
