import os
import re
import logging

import numpy as np
import iminuit
from pypescript import BasePipeline, syntax
from pypescript.config import ConfigError

from cosmopipe import section_names
from cosmopipe.lib.samples import Profiles
from cosmopipe.lib import mpi, utils, setup_logging


class MinuitProfiler(BasePipeline):

    logger = logging.getLogger('MinuitProfiler')

    def setup(self):
        self.migrad_params = self.options.get('migrad',{})
        self.minos_params = self.options.get('minos',{})
        self.torun = self.options.get('torun',None)
        if self.torun is None:
            self.torun = []
            for torun in ['migrad','minos']:
                if getattr(self,'{}_params'.format(torun)):
                    self.torun.append(torun)
        self.n_iterations = self.migrad_params.pop('n_iterations',None)
        self.max_tries = self.options.get('max_tries',1000)
        self.save = self.options.get('save',False)
        self.seed = self.options.get('seed',None)

    def execute(self):
        super(MinuitProfiler,self).setup()
        minuit_params = {}
        self.parameters = self.pipe_block[section_names.parameters,'list']
        minuit_params['name'] = parameter_names = [str(param.name) for param in self.parameters]
        self._data_block = self.data_block.copy().mpi_distribute(dests=range(self.mpicomm.size),mpicomm=mpi.COMM_SELF)

        minuit = iminuit.Minuit(self.chi2,**dict(zip(parameter_names,[param.value for param in self.parameters])),**minuit_params)
        minuit.errordef = 1.0
        for key,val in get_minuit_fixed(self.parameters).items(): minuit.fixed[key] = val
        for key,val in get_minuit_limits(self.parameters).items(): minuit.limits[key] = val
        for key,val in get_minuit_errors(self.parameters).items(): minuit.errors[key] = val

        profiles = Profiles(parameters=self.parameters)
        if 'migrad' in self.torun:
            if self.n_iterations is None:
                self.n_iterations = self.mpicomm.size
            seeds = mpi.bcast_seed(self.seed,mpicomm=self.mpicomm,size=self.n_iterations)

            def get_result(seed):
                np.random.seed(seed)
                correct_init = False
                itry = 0
                while itry < self.max_tries:
                    values = get_minuit_values(self.parameters,sample=True)
                    for key,value in zip(parameter_names,values):
                        minuit.values[key] = value
                    itry += 1
                    if np.isfinite(self.chi2(*values)):
                        correct_init = True
                        break
                if not correct_init:
                    raise ValueError('Could not find finite log posterior after {:d} calls'.format(self.max_tries))
                result = {}
                result['init'] = {par:minuit.values[par] for par in parameter_names}
                minuit.migrad(**self.migrad_params)
                loglkl = self.pipe_block[section_names.likelihood,'loglkl']
                logposterior = minuit.fval/(-2.)
                logprior = logposterior - loglkl
                result['metrics'] = {'fval':minuit.fval,'loglkl':loglkl,'logprior':logprior,'logposterior':logposterior}
                result['bestfit'] = {par:minuit.values[par] for par in parameter_names}
                result['parabolic_errors'] = {par:minuit.errors[par] for par in parameter_names}
                result['covariance'] = np.array(minuit.covariance)
                return result

            with utils.TaskManager(nprocs_per_task=1) as tm:
                results = tm.map(get_result,seeds)

            for name in ['metrics','init','bestfit','parabolic_errors']:
                getattr(profiles,'set_{}'.format(name))({key: np.array([res[name][key] for res in results]) for key in results[0][name]})

            ibest = profiles.argmin()
            profiles.set_covariance(results[ibest]['covariance'])

        #if self.profiles_key:
        #    profiles = self.data_block[self.profiles_key].append(profiles)

        if 'minos' in self.torun:

            if 'migrad' not in self.torun:
                if not hasattr('profiles','bestfit'):
                    raise RuntimeError('Migrad should be run before minos!')

            ibest = profiles.argmin()
            for param,val in profiles.bestfit.items(): minuit.values[str(param)] = val[ibest]

            if 'parameters' in self.minos_params:
                parameters = self.minos_params.pop('parameters')
            else:
                parameters = [par for par in parameter_names if not minuit.fixed[par]]

            values = []
            def get_errors(param):
                minuit.minos(param,**self.minos_params)
                return (minuit.merrors[param].lower,minuit.merrors[param].upper)

            with utils.TaskManager(nprocs_per_task=1) as tm:
                values = tm.map(get_errors,parameters)

            profiles.set_deltachi2_errors({param:value for param,value in zip(parameters,values)})

        self.data_block[section_names.likelihood,'profiles'] = profiles
        if self.save: profiles.save_auto(self.save)

    def chi2(self, *values):
        logprior = 0
        self.pipe_block = self._data_block.copy()
        for param,value in zip(self.parameters,values):
            logprior += param.prior(value)
        if np.isinf(logprior):
            return logprior
        for param,value in zip(self.parameters,values):
            self.pipe_block[param.name.tuple] = value
        for todo in self.execute_todos:
            todo()
        return -2.*(self.pipe_block[section_names.likelihood,'loglkl'] + logprior)

    def cleanup(self):
        pass


def get_minuit_values(parameters, sample=True):
    toret = []
    for param in parameters:
        name = str(param.name)
        if sample and param.varied and param.ref.is_proper():
            toret.append(param.ref.sample())
        else:
            toret.append(param.value)
    return toret


def get_minuit_fixed(parameters):
    toret = {}
    for param in parameters:
        toret[str(param.name)] = not param.varied
    return toret


def get_minuit_errors(parameters):
    toret = {}
    for param in parameters:
        if hasattr(param.ref,'scale'):
            toret[str(param.name)] = param.ref.scale
        elif param.ref.is_proper():
            toret[str(param.name)] = np.diff(param.ref.limits)/2.
    return toret


def get_minuit_limits(parameters):
    toret = {}
    for param in parameters:
        toret[str(param.name)] = tuple(None if np.isinf(lim) else lim for lim in param.prior.limits)
    return toret
