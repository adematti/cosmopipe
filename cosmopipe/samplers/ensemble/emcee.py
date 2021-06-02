import logging

import emcee
from cosmopipe.lib import Samples

from .ensemble import EnsembleSampler


class EmceeSampler(EnsembleSampler):

    logger = logging.getLogger('EmceeSampler')

    def init_sampler(self):
        self.sampler = emcee.EnsembleSampler(self.nwalkers,len(self.varied),self.logposterior,pool=self.pool)
        # TODO: make it less hacky
        import numpy as np
        self.sampler._random = np.random

    def sample(self, start):
        for _ in self.sampler.sample(initial_state=start,iterations=self.check_every,progress=False,store=True,thin_by=self.thin_by):
            pass
        chain = self.sampler.get_chain()
        data = {param.name:chain[...,iparam].flatten() for iparam,param in enumerate(self.varied)}
        data['metrics.logposterior'] = self.sampler.get_log_prob().flatten()
        self.sampler.reset()
        return Samples(data=data,parameters=self.varied,mpicomm=self.mpicomm,mpiroot=0,mpistate='gathered')
