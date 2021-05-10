import os
import re
import sys
import logging

import numpy as np
from pypescript import BasePipeline

import cobaya
cobaya.log.exception_handler = sys.excepthook # deactivate cobaya's exception handler
from cobaya.model import get_model
from cobaya.sampler import get_sampler
from cobaya.mpi import is_main_process, sync_processes
from cobaya.log import LoggedError

from cosmopipe import section_names
from cosmopipe.lib import Samples, mpi, utils, setup_logging
from .likelihood import CosmopipeLikelihood


class CobayaSampler(BasePipeline):

    logger = logging.getLogger('CobayaSampler')

    def setup(self):
        #super(CobayaSampler,self).setup()
        self.cobaya_likelihood_name = '{}_like'.format(self.options.get_string('likelihood_name','cosmopipe'))
        self.cobaya_sampler_name = list(self.options.get('sampler').keys())[0]
        self.cobaya_info_sampler = self.options['sampler']
        self.cobaya_requirements = self.options.get_list('requirements',[])
        self.seed = self.options.get('seed',None)
        self.save = self.options.get('save',False)
        self.save_cosmomc = self.options.get('save_cosmomc',False)
        self.extra_output = self.options.get_string('extra_output','').split()

    def execute(self):
        super(CobayaSampler,self).setup()
        cobaya.mpi._mpi_comm = self.mpicomm

        #cobaya.theory.always_stop_exceptions = (Exception,)
        #from cobaya.mpi import get_mpi_comm
        #print(self.mpicomm,get_mpi_comm())
        self.cobaya_params, self.cosmopipe_params = {},{}
        for param in self.pipe_block[section_names.parameters,'list']:
            name = str(param.name)
            param = get_cobaya_parameter(param)
            if (name in CosmopipeLikelihood.renames) and self.cobaya_requirements:
                self.cobaya_params[CosmopipeLikelihood.cosmopipe_to_cobaya_name(name)] = param
            else:
                self.cosmopipe_params[name] = param
                self.cobaya_params[name] = param

        likelihood = get_cosmopipe_likelihood(self,self.cosmopipe_params)
        #if 'rdrag' in likelihood().get_requirements():
        #    self.cobaya_params['rdrag'] = {'latex': 'r_\mathrm{drag}'}
        info = {}
        info['params'] = self.cobaya_params
        info['likelihood'] = {self.cobaya_likelihood_name:likelihood}
        info['likelihood'].update(self.options.get('likelihood',{}))
        info['theory'] = self.options.get('theory',{})

        if self.cobaya_sampler_name == 'mcmc':
            mpi.set_independent_seed(seed=self.seed,mpicomm=self.mpicomm)
        else:
            mpi.set_common_seed(seed=self.seed,mpicomm=self.mpicomm)
        self._data_block = self.data_block
        self.data_block = BasePipeline.mpi_distribute(self.data_block.copy(),dests=range(self.mpicomm.size),mpicomm=mpi.COMM_SELF)

        success = False
        try:
            cobaya_model = get_model(info)
            sampler = get_sampler(self.cobaya_info_sampler,model=cobaya_model)
            sampler.run()
            success = True
        except LoggedError as err:
            pass
        success = all(self.mpicomm.allgather(success))
        if not success:
            self.log_warning('Sampling failed!',rank=0)

        self.mpicomm.Barrier()
        self.data_block = self._data_block
        samples = Samples(parameters=self.pipe_block[section_names.parameters,'list'],mpicomm=self.mpicomm)

        if self.cobaya_sampler_name == 'mcmc':
            output = sampler.products()['sample']
            # multiple chains run in parallel, we interleave them
            samples.attrs['interleaved_chain_sizes'] = sizes = self.mpicomm.allgather(output.values.shape[0])
            values = mpi.gather_array(output.values,mpicomm=self.mpicomm,root=0)
            if self.mpicomm.rank == 0:
                csizes = [0] + np.cumsum(sizes).tolist()
                values = utils.interleave(*[values[start:stop] for start,stop in zip(csizes[:-1],csizes[1:])])
        elif self.mpicomm.rank == 0:
            output = sampler.products()['sample']
            values = output.values

        if self.mpicomm.rank == 0:
            convert_minus = {'minuslogprior':('metrics','logprior'),'minuslogpost':('metrics','logposterior')}
            for col,val in zip(output.columns,values.T):
                match = re.match('chi2__(.*)_like$',col)
                name = CosmopipeLikelihood.cobaya_to_cosmopipe_name(col)
                if match:
                    samples['metrics','loglkl_{}'.format(match.group(1))] = -1./2*val
                elif col in convert_minus:
                    samples[convert_minus[col]] = -val
                elif col == 'weight':
                    samples['metrics','weight'] = val
                elif name is not None:
                    samples[name] = val
            samples = samples[~np.isnan(samples['metrics','weight'])]
            if self.save: samples.save(self.save)
            if self.save_cosmomc: samples.save_cosmomc(self.save_cosmomc)

        samples.mpi_scatter()
        self.data_block[section_names.likelihood,'samples'] = samples


def get_cosmopipe_likelihood(pipeline, params):

    # create class copy, otherwise following changes will be conserved in the Python run
    cls = type('_CosmopipeLikelihood', CosmopipeLikelihood.__bases__, dict(CosmopipeLikelihood.__dict__))
    cls._pipeline = pipeline
    cls.params = params

    def initialize(self, *args, **kwargs):
        pass

    cls.requirements = pipeline.cobaya_requirements
    cls.initialize = initialize
    return cls


def get_cobaya_parameter(parameter):

    if parameter.fixed:
        return parameter.value
    toret = {}
    toret['latex'] = parameter.latex
    toret['proposal'] = parameter.proposal

    for attr in ['prior','ref']:
        dist = getattr(parameter,attr)
        toret[attr] = {}
        toret[attr]['dist'] = dist.dist
        for key in dist._keys:
            toret[attr][key] = getattr(dist,key)
        if dist._keys:
            if dist.is_limited():
                parameter.logger.warning('In {} {} for parameter {} '\
                'Cobaya does not accept limits = {} when loc/scale are provided. Dropping limits.'.format(dist.dist,attr,parameter.name,parameter.limits))
        else:
            toret[attr] = {'min':dist.limits[0],'max':dist.limits[1]}
    return toret
