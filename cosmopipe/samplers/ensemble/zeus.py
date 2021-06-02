import logging

import zeus
from cosmopipe.lib import Samples

from .ensemble import EnsembleSampler


class ZeusSampler(EnsembleSampler):

    logger = logging.getLogger('ZeusSampler')

    def setup(self):
        super(ZeusSampler,self).setup()
        self.light_mode = self.options.get_bool('light_mode',False)

    def init_sampler(self):
        handlers = logging.root.handlers.copy()
        level = logging.root.level
        self.sampler = zeus.EnsembleSampler(self.nwalkers,len(self.varied),self.logposterior,pool=self.pool,verbose=False,light_mode=self.light_mode)
        logging.root.handlers = handlers
        logging.root.level = level

    def sample(self, start):
        for _ in self.sampler.sample(start=start,iterations=self.check_every,progress=False,thin_by=self.thin_by):
            pass
        data = self.sampler.get_chain()
        data = {param.name:data[...,iparam].flatten() for iparam,param in enumerate(self.varied)}
        data['metrics.logposterior'] = self.sampler.get_log_prob().flatten()
        self.sampler.reset()
        return Samples(data=data,parameters=self.varied,mpicomm=self.mpicomm,mpiroot=0,mpistate='gathered')
