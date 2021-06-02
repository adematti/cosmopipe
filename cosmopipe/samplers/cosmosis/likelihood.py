from cosmosis.datablock import names as cosmosis_names
from cosmosis.datablock import SectionOptions
from cosmosis.runtime.module import Module as CosmosisModule
from pypescript import BasePipeline

from cosmopipe import section_names
from cosmopipe.lib import mpi


class CosmopipeModule(CosmosisModule):

    def setup(self, config, quiet=True):
        pass

    def execute(self, data_block):
        self._pipeline.pipe_block = self._pipeline.data_block.copy()
        for param in self.parameters:
            self._pipeline.pipe_block[param.name.tuple] = data_block[param.name.tuple]
        for todo in self._pipeline.execute_todos:
            todo()
        data_block[cosmosis_names.likelihoods,'{}_like'.format(self.name)] = self._pipeline.pipe_block[section_names.likelihood,'loglkl']
        return 0

    def cleanup(self):
        pass


def setup(options):
    like = object.__new__(CosmopipeModule)
    like._pipeline = BasePipeline(config_block=SectionOptions(options).get_string('config_file'))
    like._pipeline.setup()
    like._pipeline.data_block = BasePipeline.mpi_distribute(like._pipeline.data_block,dests=like._pipeline.mpicomm.rank,mpicomm=mpi.COMM_SELF)
    like.parameters = like._pipeline.data_block[section_names.parameters,'list'] = like._pipeline.pipe_block[section_names.parameters,'list']
    like.name = 'cosmopipe'
    return like

def execute(block, config):
    like = config
    like.execute(block)
    return 0

def cleanup(config):
    like = config
    like._pipeline.cleanup()
