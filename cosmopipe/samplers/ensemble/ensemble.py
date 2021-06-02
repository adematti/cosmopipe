import sys
import logging

import numpy as np
from pypescript import BasePipeline

from cosmopipe import section_names
from cosmopipe.lib import Samples, mpi


class EnsembleSampler(BasePipeline):

    logger = logging.getLogger('EnsembleSampler')

    def setup(self):
        self.save = self.options.get('save',False)
        self.load = self.options.get('load',False)
        self.seed = self.options.get('seed',None)
        self.nwalkers = self.options.get('nwalkers',None)
        self.thin_by = self.options.get('thin_by',1)
        self.check_every = self.options.get('check_every',100)
        self.min_iterations = self.options.get('min_iterations',0)
        self.max_iterations = self.options.get('max_iterations',sys.maxsize)
        self.max_tries = self.options.get('max_tries',1000)
        diagnostics_options = ['nslabs','stable_over','burnin','eigen_gr_stop','diag_gr_stop','cl_diag_gr_stop','nsigmas_cl_diag_gr_stop','iact_stop','dact_stop']
        self.diagnostics = {key:self.options.get(key) for key in diagnostics_options if key in self.options}

    def execute(self):
        super(EnsembleSampler,self).setup()
        self._data_block = self.data_block.copy().mpi_distribute(dests=range(self.mpicomm.size),mpicomm=mpi.COMM_SELF)

        self.pool = None
        if self.mpicomm.size > 1:
            self.pool = mpi.MPIPool(mpicomm=self.mpicomm,check_tasks=True)

        parameters = self.pipe_block[section_names.parameters,'list']
        self.varied = parameters.select(fixed=False)
        self.log_info('Varying parameters {}.'.format([str(param.name) for param in self.varied]),rank=0)
        self.fixed = parameters.select(fixed=True)

        if self.nwalkers is None:
            self.nwalkers = 2 * ((int(2.5 * len(self.varied)) + 1)//2)
        self.init_sampler()

        samples = Samples(parameters=parameters,attrs={'nwalkers':self.nwalkers},mpicomm=self.mpicomm,mpiroot=0,mpistate='gathered')
        start = None

        if self.load:
            if self.mpicomm.rank == 0:
                samples = Samples.load_auto(self.load)
                start = np.array([samples[param][-self.nwalkers:] for param in self.varied]).T

        else:
            if self.mpicomm.rank == 0:
                def get_start():
                    toret = []
                    for param in self.varied:
                        if param.ref.is_proper():
                            toret.append(param.ref.sample())
                        else:
                            toret.append(param.value)
                    return toret

                start = []
                for iwalker in range(self.nwalkers):
                    correct_init = False
                    itry = 0
                    while itry < self.max_tries:
                        values = get_start()
                        itry += 1
                        if np.isfinite(self.logposterior(values)):
                            correct_init = True
                            break
                    if not correct_init:
                        raise ValueError('Could not find finite log posterior after {:d} tries'.format(self.max_tries))
                    start.append(values)
                start = np.array(start)

        start = self.mpicomm.bcast(start,root=0)
        mpi.set_common_seed(seed=self.seed,mpicomm=self.mpicomm)

        count_iterations = 0
        is_converged = False
        self._current_diagnostics = {}
        #is_converged = True
        while not is_converged:
            new = self.sample(start)
            count_iterations += self.check_every
            if self.mpicomm.rank == 0:
                for param in self.fixed: new[param] = new.full(param.value)
            samples.extend(new)
            self.log_info('Sampling has run for {:d} iterations.'.format(count_iterations),rank=0)
            is_converged = self.run_diagnostics(samples,**self.diagnostics)
            #is_converged = True
            if self.save: samples.save_auto(self.save)
            start = np.array([new[param][-self.nwalkers:] for param in self.varied]).T
            if count_iterations < self.min_iterations:
                is_converged = False
            if count_iterations > self.max_iterations:
                is_converged = True

        self.mpicomm.Barrier()
        samples.mpi_scatter()
        self.data_block[section_names.likelihood,'samples'] = samples

    def init_sampler(self):
        raise NotImplementedError

    def sample(self, start):
        raise NotImplementedError

    def logposterior(self, values):
        logprior = 0
        for param,value in zip(self.varied,values):
            logprior += param.prior(value)
        if np.isinf(logprior):
            return logprior
        self.pipe_block = self._data_block.copy()
        for param,value in zip(self.varied,values):
            self.pipe_block[param.name.tuple] = value
        for param in self.fixed:
            self.pipe_block[param.name.tuple] = param.value
        for todo in self.execute_todos:
            todo()
        return self.pipe_block[section_names.likelihood,'loglkl'] + logprior

    def run_diagnostics(self, samples, nslabs=4, stable_over=2, burnin=0.3, eigen_gr_stop=0.03, diag_gr_stop=None,
                        cl_diag_gr_stop=None, nsigmas_cl_diag_gr_stop=1., iact_stop=None, iact_reliable=50, dact_stop=None):

        def add_diagnostics(name, value):
            if name not in self._current_diagnostics:
                self._current_diagnostics[name] = [value]
            else:
                self._current_diagnostics[name].append(value)
            return value

        def is_stable(name):
            if len(self._current_diagnostics[name]) < stable_over:
                return False
            return all(self._current_diagnostics[name][-stable_over:])

        if 0 < burnin < 1:
            burnin = burnin*(samples.gsize//self.nwalkers)
        burnin = int(round(burnin))*self.nwalkers

        lenslabs = (samples.gsize - burnin) // nslabs

        msg = '{{}} is {{:.3g}}'.format(samples.gsize//self.nwalkers)

        if lenslabs < 2:
            return False

        chains = [samples.gslice(burnin + islab*lenslabs, burnin + (islab + 1)*lenslabs) for islab in range(nslabs)]

        self.log_info('Diagnostics:',rank=0)
        item = '- '
        toret = True

        eigen_gr = Samples.gelman_rubin(chains,self.varied,method='eigen',check=False).max() - 1
        msg = '{}max eigen Gelman-Rubin - 1 is {:.3g}'.format(item,eigen_gr)
        if eigen_gr_stop is not None:
            test = eigen_gr < eigen_gr_stop
            self.log_info('{} {} {:.3g}.'.format(msg,'<' if test else '>',eigen_gr_stop),rank=0)
            add_diagnostics('eigen_gr',test)
            toret = is_stable('eigen_gr')
        else:
            self.log_info('{}.'.format(msg),rank=0)

        diag_gr = Samples.gelman_rubin(chains,self.varied,method='diag').max() - 1
        msg = '{}max diag Gelman-Rubin - 1 is {:.3g}'.format(item,diag_gr)
        if diag_gr_stop is not None:
            test = diag_gr < diag_gr_stop
            self.log_info('{} {} {:.3g}.'.format(msg,'<' if test else '>',diag_gr_stop),rank=0)
            add_diagnostics('diag_gr',test)
            toret = is_stable('diag_gr')
        else:
            self.log_info('{}.'.format(msg),rank=0)

        def cl_lower(samples, parameters):
            return samples.interval(parameters,nsigmas=nsigmas_cl_diag_gr_stop)[:,0]

        def cl_upper(samples, parameters):
            return samples.interval(parameters,nsigmas=nsigmas_cl_diag_gr_stop)[:,1]

        cl_diag_gr = np.max([Samples.gelman_rubin(chains,self.varied,statistic=cl_lower,method='diag'),
                        Samples.gelman_rubin(chains,self.varied,statistic=cl_upper,method='diag')]) - 1
        msg = '{}max diag Gelman-Rubin - 1 at {:.1f} sigmas is {:.3g}'.format(item,nsigmas_cl_diag_gr_stop,cl_diag_gr)
        if cl_diag_gr_stop is not None:
            test = cl_diag_gr - 1 < cl_diag_gr_stop
            self.log_info('{} {} {:.3g}'.format(msg,'<' if test else '>',cl_diag_gr_stop),rank=0)
            add_diagnostics('cl_diag_gr',test)
            toret = is_stable('cl_diag_gr')
        else:
            self.log_info('{}.'.format(msg),rank=0)

        chains = [samples.gslice(burnin + iwalker,None,self.nwalkers) for iwalker in range(self.nwalkers)]
        iact = Samples.integrated_autocorrelation_time(chains,self.varied)
        add_diagnostics('tau',iact)

        iact = iact.max()
        msg = '{}max integrated autocorrelation time is {:.3g}'.format(item,iact)
        n_iterations = chains[0].gsize
        if iact_reliable * iact < n_iterations:
            msg = '{} (reliable)'.format(msg)
        if iact_stop is not None:
            test = iact * iact_stop < n_iterations
            self.log_info('{} {} {:d}/{:.1f} = {:.3g}'.format(msg,'<' if test else '>',n_iterations,iact_stop,n_iterations/iact_stop),rank=0)
            add_diagnostics('iact',test)
            toret = is_stable('iact')
        else:
            self.log_info('{}.'.format(msg),rank=0)

        tau = self._current_diagnostics['tau']
        if len(tau) >= 2:
            rel = np.abs(tau[-2]/tau[-1] - 1).max()
            msg = '{}max variation of integrated autocorrelation time is {:.3g}'.format(item,rel)
            if dact_stop is not None:
                test = rel < dact_stop
                self.log_info('{} {} {:.3g}'.format(msg,'<' if test else '>',dact_stop),rank=0)
                add_diagnostics('dact',test)
                toret = is_stable('dact')
            else:
                self.log_info('{}.'.format(msg),rank=0)

        return toret
