import os
import re
import logging

import numpy as np
import iminuit
from pypescript import BasePipeline
from pypescript.config import ConfigError

from cosmopipe import section_names
from cosmopipe.lib.samples import Profiles
from cosmopipe.lib import mpi, utils, setup_logging


class MinuitProfiler(BasePipeline):

    logger = logging.getLogger('Minuit')

    def setup(self):
        self.minuit_params = self.options.get('minuit',{})
        self.migrad_params = self.options.get('migrad',{})
        self.minos_params = self.options.get('minos',{})
        self.iter = self.options.get('iter',None)
        self.torun = self.options.get('torun',['migrad'])
        self.save = self.options.get('save',False)
        self.seed = self.options.get('seed',None)
        self.profiles_key = self.options.get('profiles_key',None)
        if self.profiles_key is not None:
            self.profiles_key = utils.split_section_name(self.profiles_key)
            if len(self.profiles_key) == 1:
                self.profiles_key = (section_names.likelihood,) + self.profiles_key

    def execute(self):
        super(MinuitProfiler,self).setup()
        minuit_params = self.minuit_params.copy()
        self.parameter_names = []
        parameters = self.pipe_block[section_names.parameters,'list']
        for param in parameters:
            self.parameter_names.append(str(param.name))
        minuit_params['name'] = self.parameter_names
        self._data_block = self.data_block
        self.data_block = BasePipeline.mpi_distribute(self.data_block.copy(),dests=self.mpicomm.rank,mpicomm=mpi.COMM_SELF)
        chi2 = get_cosmopipe_chi2(self)
        minuit = iminuit.Minuit(chi2,**get_minuit_values(parameters,sample=False),**minuit_params)
        minuit.errordef = 1.0
        for key,val in get_minuit_fixed(parameters).items(): minuit.fixed[key] = val
        for key,val in get_minuit_limits(parameters).items(): minuit.limits[key] = val
        for key,val in get_minuit_errors(parameters).items(): minuit.errors[key] = val

        profiles = Profiles(parameters=parameters)
        if 'migrad' in self.torun:
            if self.iter is None:
                self.iter = self.mpicomm.size
            seeds = mpi.bcast_seed(self.seed,mpicomm=self.mpicomm)[:self.iter]

            def get_result(seed):
                np.random.seed(seed)
                for key,val in get_minuit_values(parameters,sample=True).items(): minuit.values[key] = val
                result = {}
                result['init'] = {par:minuit.values[par] for par in self.parameter_names}
                minuit.migrad(**self.migrad_params)
                result['metrics'] = {'minchi2':minuit.fval}
                result['bestfit'] = {par:minuit.values[par] for par in self.parameter_names}
                result['parabolic_errors'] = {par:minuit.errors[par] for par in self.parameter_names}
                result['covariance'] = np.array(minuit.covariance)
                return result

            with utils.TaskManager(nprocs_per_task=1) as tm:
                results = tm.map(get_result,seeds)

            for name in ['metrics','init','bestfit','parabolic_errors']:
                getattr(profiles,'set_{}'.format(name))({key: np.array([res[name][key] for res in results]) for key in results[0][name]})

            ibest = profiles.argmin()
            profiles.set_covariance(results[ibest]['covariance'])

        if self.profiles_key:
            profiles = self.data_block[self.profiles_key].append(profiles)

        if 'minos' in self.torun:

            if 'migrad' not in self.torun:
                if not hasattr('profiles','bestfit'):
                    raise RuntimeError('Migrad should be run before minos!')

            ibest = profiles.argmin()
            for param,val in profiles.bestfit.items(): minuit.values[str(param)] = val[ibest]

            if 'parameters' in self.minos_params:
                parameters = self.minos_params.pop('parameters')
            else:
                parameters = [par for par in self.parameter_names if not minuit.fixed[par]]

            values = []
            def get_errors(param):
                minuit.minos(param,**self.minos_params)
                return (minuit.merrors[param].lower,minuit.merrors[param].upper)

            with utils.TaskManager(nprocs_per_task=1) as tm:
                values = tm.map(get_errors,parameters)

            profiles.set_deltachi2_errors({param:value for param,value in zip(parameters,values)})

        self.data_block = self._data_block
        self.data_block[section_names.likelihood,'profiles'] = profiles
        if self.save:
            profiles.save(self.save)

    def cleanup(self):
        pass


def get_minuit_values(parameters, sample=True):
    toret = {}
    for param in parameters:
        name = str(param.name)
        toret[name] = param.value
        if sample and (not param.fixed) and param.ref.is_proper():
            toret[name] = param.ref.sample()
    return toret


def get_minuit_fixed(parameters):
    toret = {}
    for param in parameters:
        toret[str(param.name)] = param.fixed
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


def get_cosmopipe_chi2(self):

    def chi2(*args):
        self.pipe_block = self.data_block.copy()
        for iparam,param in enumerate(self.pipe_block[section_names.parameters,'list']):
            self.pipe_block[param.name.tuple] = args[iparam]
        for todo in self.execute_todos:
            todo()
        return -2.*self.pipe_block[section_names.likelihood,'loglkl']

    return chi2
