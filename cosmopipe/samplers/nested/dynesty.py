import logging

import numpy as np
import dynesty

from cosmopipe import section_names
from cosmopipe.lib import Samples, mpi, ParamError

from pypescript import BasePipeline


class DynestySampler(BasePipeline):

    logger = logging.getLogger('DynestySampler')

    def setup(self):
        self.mode = self.options.get('mode','static')
        self.nlive = self.options.get('nlive',500)
        self.bound = self.options.get('bound','multi')
        self.sample = self.options.get('sample','auto')
        self.update_interval = self.options.get('update_interval',None)
        #self.queue_size = self.options.get('queue_size',None)
        self.max_iterations = self.options.get('max_iterations',None)
        self.dlogz = self.options.get('dlogz',0.01)
        self.seed = self.options.get('seed',None)
        self.save = self.options.get('save',False)

        super(DynestySampler,self).setup()
        self.parameters = self.pipe_block[section_names.parameters,'list']

    def execute(self):
        self._data_block = self.data_block.copy().mpi_distribute(dests=range(self.mpicomm.size),mpicomm=mpi.COMM_SELF)

        self.pool = None
        mpi.set_common_seed(seed=self.seed,mpicomm=self.mpicomm)
        if self.mpicomm.size > 1:
            self.pool = mpi.MPIPool(mpicomm=self.mpicomm,check_tasks=True)
            self.queue_size = self.pool.size

        self.varied = self.parameters.select(varied=True)
        self.log_info('Varying parameters {}.'.format([str(param.name) for param in self.varied]),rank=0)
        for param in self.varied:
            if not param.prior.is_proper():
                raise ParamError('Prior for {} is improper, Dynesty requires proper priors'.format(param.name))
        self.fixed = self.parameters.select(varied=False)

        ndim = len(self.varied)
        # e.g. propose_point requires the whole lkl to be pickelable...
        use_pool = {'prior_transform':False,'loglikelihood':True,'propose_point':False,'update_bound':False}

        if self.mode == 'static':
            sampler = dynesty.NestedSampler(
                self.loglkl,
                self.prior_transform,
                ndim,
                nlive=self.nlive,
                bound=self.bound,
                sample=self.sample,
                update_interval=self.update_interval,
                #queue_size=self.queue_size,
                pool=self.pool,
                use_pool=use_pool,
                )

            sampler.run_nested(maxiter=self.max_iterations,dlogz=self.dlogz)

        else:
            sampler = dynesty.DynamicNestedSampler(
                self.loglkl,
                self.prior_transform,
                ndim,
                bound=self.bound,
                sample=self.sample,
                update_interval=self.update_interval,
                #queue_size=self.queue_size,
                pool=self.pool,
                use_pool=use_pool,
                )
            sampler.run_nested(nlive_init=self.nlive,maxiter=self.max_iterations,dlogz_init=self.dlogz)

        results = sampler.results

        samples = Samples(parameters=self.parameters,mpicomm=self.mpicomm,mpiroot=0,mpistate='gathered')
        data = {param.name:results['samples'][...,iparam] for iparam,param in enumerate(self.varied)}
        data['metrics.loglkl'] = results['logl']
        # TODO: to compare to logvol!
        logprior = 0
        for param in self.varied: logprior += param.prior(data[param.name])
        data['metrics.logprior'] = logprior
        data['metrics.logposterior'] = results['logl'] + data['metrics.logprior']
        data['metrics.logweight'] = results['logwt']
        data['metrics.fweight'] = np.exp(results.logwt - results.logz[-1])
        new = Samples(data=data,parameters=self.varied,mpicomm=self.mpicomm,mpiroot=0,mpistate='gathered')
        for param in self.fixed: new[param] = new.full(param.value)
        samples.extend(new)
        for key in ['ncall','eff','logz','logzerr']:
            samples.attrs[key] = results[key]
        if self.save: samples.save_auto(self.save)

        self.mpicomm.Barrier()

        samples.mpi_scatter()
        self.data_block[section_names.likelihood,'samples'] = samples

    def loglkl(self, values):
        self.pipe_block = self.data_block.copy()
        for param,value in zip(self.varied,values):
            self.pipe_block[param.name.tuple] = value
        for param in self.fixed:
            self.pipe_block[param.name.tuple] = param.value
        for todo in self.execute_todos:
            todo()
        return self.pipe_block[section_names.likelihood,'loglkl']

    def prior_transform(self, values):
        toret = np.empty_like(values)
        for iparam,(value,param) in enumerate(zip(values,self.varied)):
             toret[iparam] = param.prior.ppf(value)
        return toret
