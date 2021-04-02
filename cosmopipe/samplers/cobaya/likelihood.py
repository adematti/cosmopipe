from cobaya.likelihood import Likelihood as CobayaLikelihood
from pypescript import BasePipeline

from cosmopipe import section_names
from cosmopipe.lib import mpi


class CosmopipeLikelihood(CobayaLikelihood):

    config_file = 'config.yaml'
    #param_file = 'param.yaml'

    def initialize(self):
        self._pipeline = BasePipeline(config_block=self.config_file)
        self._pipeline.setup()
        self._pipeline.data_block = BasePipeline.mpi_distribute(self._pipeline.data_block,dests=self._pipeline.mpicomm.rank,mpicomm=mpi.COMM_SELF)
        from .sampler import get_cobaya_parameter
        self.params = {str(param.name):param.value for param in self._pipeline.pipe_block[section_names.parameters,'list']}

    def get_requirements(self):
        return {}

    def logp(self, **kwargs):
        self._pipeline.pipe_block = self._pipeline.data_block.copy()
        for param in self._pipeline.pipe_block[section_names.parameters,'list']:
            self._pipeline.pipe_block[param.name.tuple] = kwargs[str(param.name)]
        for todo in self._pipeline.execute_todos:
            todo()
        return self._pipeline.pipe_block[section_names.likelihood,'loglkl']

    def clean(self):
        pass
