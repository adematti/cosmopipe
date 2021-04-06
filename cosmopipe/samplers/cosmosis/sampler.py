import os
import re
import logging
import sys

import numpy as np
from cosmosis.runtime.config import Inifile
from cosmosis.runtime.pipeline import Pipeline as CosmosisPipeline
from cosmosis.runtime.pipeline import LikelihoodPipeline as CosmosisLikelihoodPipeline
from cosmosis.runtime.pipeline import PIPELINE_INI_SECTION as cosmosis_pipeline_name
from cosmosis.runtime.parameter import Parameter as CosmosisParameter
#from cosmosis.samplers.sampler import ParallelSampler
#from cosmosis.runtime.prior import Prior as CosmosisPrior
from cosmosis.output.in_memory_output import InMemoryOutput
from cosmosis.samplers.sampler import Sampler
from cosmosis.datablock import names as cosmosis_names
from pypescript import BasePipeline

from cosmopipe import section_names
from cosmopipe.lib import Samples, mpi, utils
from .likelihood import CosmopipeModule


class CosmosisSampler(BasePipeline):

    logger = logging.getLogger('CosmosisSampler')

    def setup(self):
        self.cosmosis_likelihood_name = self.options.get_string('cosmosis_likelihood_name','cosmopipe')
        self.cosmosis_sampler_name = self.options['sampler']
        self.cosmosis_sampler_class = Sampler.registry[self.cosmosis_sampler_name]
        self.seed = self.options.get('seed',None)
        self.save = self.options.get('save',False)
        self.save_cosmomc = self.options.get('save_cosmomc',False)
        self.extra_output = self.options.get_string('extra_output','').split()
        exclude = self._reserved_option_names + ['sampler','config_cosmosis','extra_output','save','save_cosmomc']
        override = {(self.cosmosis_sampler_name,name):str(value) for name,value in self.options.items() if name not in exclude}
        override[cosmosis_pipeline_name,'modules'] = ''
        override[cosmosis_pipeline_name,'quiet'] = 'T'
        override[cosmosis_pipeline_name,'debug'] = 'F'
        override[cosmosis_pipeline_name,'timing'] = 'F'
        if self.options.has('config_cosmosis'):
            ini = Inifile(self.options.get_string('config_cosmosis'))
            override.update(**ini)
        self.cosmosis_ini = Inifile(filename=None,override=override)

    def execute(self):
        super(CosmosisSampler,self).setup()
        self.cosmosis_pipeline = get_cosmosis_likelihood_pipeline(self)
        output = InMemoryOutput()
        pool = False
        kwargs = {}
        if self.cosmosis_sampler_class.is_parallel_sampler:
            mpi.set_common_seed(seed=self.seed,mpicomm=self.mpicomm)

            if self.mpicomm.size > 1:
                    kwargs['pool'] = pool = mpi.MPIPool(mpicomm=self.mpicomm)

            def is_master(self):
                return True

            self.cosmosis_sampler_class.is_master = is_master # hack to get sampler running on all ranks
        else:
            mpi.set_independent_seed(seed=self.seed,mpicomm=self.mpicomm)

        sampler = self.cosmosis_sampler_class(self.cosmosis_ini,self.cosmosis_pipeline,output,**kwargs)
        sampler.config()
        self._data_block = self.data_block
        self.data_block = BasePipeline.mpi_distribute(self.data_block.copy(),dests=range(self.mpicomm.size),mpicomm=mpi.COMM_SELF)

        sampler.execute()
        #while not sampler.is_converged():
        #    sampler.execute()
        #    if output:
        #        output.flush()
        #if pool and sampler.is_parallel_sampler:
        #    pool.close()
        self.mpicomm.Barrier()
        self.data_block = self._data_block
        samples = Samples(parameters=self.pipe_block[section_names.parameters,'list'],attrs={**output.meta,**output.final_meta})

        values = np.array(output.rows)
        if not self.cosmosis_sampler_class.is_parallel_sampler:
            # multiple chains run in parallel, we interleave them
            samples.attrs['interleaved_chain_sizes'] = sizes = self.mpicomm.allgather(values.shape[0])
            values = mpi.gather_array(values,mpicomm=self.mpicomm,root=0)
            if self.mpicomm.rank == 0:
                csizes = [0] + np.cumsum(sizes).tolist()
                values = utils.interleave(*[values[start:stop] for start,stop in zip(csizes[:-1],csizes[1:])])

        if self.mpicomm.rank == 0:
            convert = {'prior':('metrics','logprior'),'post':('metrics','logposterior')}
            #if self.mpicomm.rank == 0:
            for col,val in zip(output.columns,values.T):
                col = col[0]
                match = re.match('(.*)--(.*)',col)
                if match:
                    samples[match.group(1),match.group(2)] = val
                elif col in convert:
                    samples[convert[col]] = val
            if self.save: samples.save(self.save)
            if self.save_cosmomc: samples.save_cosmomc(self.save_cosmomc)
        samples.mpi_scatter()
        self.data_block[section_names.likelihood,'samples'] = samples
        if output:
            output.close()
        self.cosmosis_pipeline.cleanup()


def get_cosmosis_likelihood_pipeline(pipeline):

    self = object.__new__(CosmosisLikelihoodPipeline)
    CosmosisPipeline.__init__(self,arg=pipeline.cosmosis_ini,load=True)
    self.modules += [get_cosmopipe_module(pipeline)]
    self.n_iterations = 0
    self.parameters = []
    values_file = pipeline.cosmosis_ini.get(cosmosis_pipeline_name,'values',fallback=None)
    if values_file is not None:
        priors_file = pipeline.cosmosis_ini.get(cosmosis_pipeline_name,'priors',fallback='')
        self.parameters += parameter.Parameter.load_parameters(values_file,priors_file)
    self.parameters += [get_cosmosis_parameter(param) for param in pipeline.data_block[section_names.parameters,'list']]
    self.reset_fixed_varied_parameters()
    self.print_priors()
    self.extra_saves = pipeline.extra_output
    self.number_extra = len(self.extra_saves)
    self.likelihood_names = [pipeline.cosmosis_likelihood_name]
    self.setup()
    return self


def get_cosmosis_parameter(parameter):

    self = object.__new__(CosmosisParameter)
    self.section = parameter.name.tuple[0]
    self.name = parameter.name.tuple[-1]
    self.start = parameter.value
    for key in ['limits','prior']:
        setattr(self,key,getattr(parameter,key))
    if parameter.fixed:
        self.limits = (parameter.value,)*2
    return self


def get_cosmopipe_module(pipeline):

    self = object.__new__(CosmopipeModule)

    def setup(*args, **kwargs):
        pass

    self.setup = setup
    self._pipeline = pipeline
    self.name = self._pipeline.cosmosis_likelihood_name
    self.filename = os.path.realpath(__file__)

    def setup(*args, **kwargs):
        pass

    def cleanup(*args, **kwargs):
        pass

    # Handled by CosmosisSampler
    self.setup = setup
    self.cleanup = cleanup

    return self
