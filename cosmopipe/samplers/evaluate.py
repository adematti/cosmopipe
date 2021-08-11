import logging

import numpy as np
from pypescript import BasePipeline

from cosmopipe import section_names
from cosmopipe.lib import mpi


class EvaluateSampler(BasePipeline):

    logger = logging.getLogger('EvaluateSampler')

    def setup(self):
        self.max_tries = self.options.get('max_tries',1000)
        super(EvaluateSampler,self).setup()
        self.parameters = self.pipe_block[section_names.parameters,'list']

    def execute(self):
        self._data_block = self.data_block.copy().mpi_distribute(dests=range(self.mpicomm.size),mpicomm=mpi.COMM_SELF)

        self.varied = self.parameters.select(varied=True)
        self.log_info('Varying parameters {}.'.format([str(param.name) for param in self.varied]),rank=0)
        self.fixed = self.parameters.select(varied=False)

        def get_start():
            toret = []
            for param in self.varied:
                if param.ref.is_proper():
                    toret.append(param.ref.sample())
                else:
                    toret.append(param.value)
            return toret

        correct_init = False
        itry,logposterior = 0,np.inf
        while itry < self.max_tries:
            values = get_start()
            logposterior = self.logposterior(values)
            itry += 1
            if np.isfinite(logposterior):
                correct_init = True
                break
        if not correct_init:
            raise ValueError('Could not find finite log posterior after {:d} tries'.format(self.max_tries))
        self.log_info('Finite log-posterior value of {:.3g} found after {:d} tries.'.format(logposterior,itry),rank=0)

        self.mpicomm.Barrier()

    def logposterior(self, values):
        logprior = 0
        for value,param in zip(values,self.varied):
            logprior += param.prior(value)
        if np.isinf(logprior):
            return logprior
        self.pipe_block = self._data_block.copy()
        for value,param in zip(values,self.varied):
            self.pipe_block[param.name.tuple] = value
        for param in self.fixed:
            self.pipe_block[param.name.tuple] = param.value
        for todo in self.execute_todos:
            todo()
        return self.pipe_block[section_names.likelihood,'loglkl'] + logprior
