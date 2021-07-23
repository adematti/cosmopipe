import logging

import numpy as np
from pypescript import BasePipeline

from cosmopipe import section_names
from cosmopipe.lib.data_vector import DataVector
from cosmopipe.lib.theory import ModelCollection


class BaseLikelihood(BasePipeline):

    logger = logging.getLogger('BaseLikelihood')

    def setup(self):
        super(BaseLikelihood,self).setup()
        self.set_data()

    def set_data(self):
        self.data = self.pipe_block[section_names.data,'y']
        #self.data_block[section_names.data] = self.pipe_block[section_names.data]

    def set_model(self):
        #self.model = self.data_block[section_names.model,'y'] = self.pipe_block[section_names.model,'y']
        self.model = self.pipe_block[section_names.model,'y']

    def loglkl(self):
        return 0

    def execute(self):
        super(BaseLikelihood,self).execute()
        self.set_model()
        self.data_block[section_names.likelihood,'loglkl'] = self.loglkl()


class GaussianLikelihood(BaseLikelihood):

    logger = logging.getLogger('GaussianLikelihood')

    def setup(self):
        super(GaussianLikelihood,self).setup()
        self.set_covariance()

    def set_covariance(self):
        self.invcovariance = self.pipe_block[section_names.covariance,'invcov']
        self.nobs = self.pipe_block.get(section_names.covariance,'nobs',None)
        if self.nobs is None:
            self.log_info('The number of observations used to estimate the covariance matrix is not provided,'\
                            ' hence no Hartlap factor is applied to inverse covariance.',rank=0)
            self.precision = self.invcovariance
        else:
            self.hartlap = (self.nobs - self.data.size - 2.)/(self.nobs - 1.)
            self.log_info('Covariance matrix with {:d} points built from {:d} observations.'.format(self.data.size,self.nobs),rank=0)
            self.log_info('...resulting in Hartlap factor of {:.4f}.'.format(self.hartlap),rank=0)
            self.precision = self.invcovariance * self.hartlap
        if np.ndim(self.precision) == 0:
            self.precision = self.precision * np.eye(self.data.size,dtype=self.data.dtype)

    def loglkl(self):
        diff = self.model - self.data
        return -0.5*diff.T.dot(self.precision).dot(diff)


class SumLikelihood(BaseLikelihood):

    logger = logging.getLogger('SumLikelihood')

    def setup(self):
        BasePipeline.setup(self)

    def execute(self):
        loglkl = 0
        self.pipe_block = self.data_block.copy()
        for todo in self.execute_todos:
            todo()
            loglkl += self.pipe_block[section_names.likelihood,'loglkl']
        self.data_block[section_names.likelihood,'loglkl'] = loglkl


class JointGaussianLikelihood(GaussianLikelihood):

    logger = logging.getLogger('JointGaussianLikelihood')

    @classmethod
    def _join_values(cls, key, value, other):
        if value is None:
            return other.copy()
        if key == (section_names.data,'data_vector'):
            return DataVector.concatenate(value,other)
        if key == (section_names.model,'collection'):
            return ModelCollection.concatenate(value,other)
        return np.concatenate([value,other],axis=0)

    @classmethod
    def _run_todos(cls, todolist, join):
        for todo in todolist:
            module = todo.module
            islike = isinstance(module,BaseLikelihood)
            for key,value in join.items():
                if value is not None:
                    if not islike:
                        todo.pipeline.pipe_block[key] = value # e.g. feed data vector to covariance matrix
                    elif key in todo.pipeline.pipe_block and todo.pipeline.pipe_block[key] is value:
                        del todo.pipeline.pipe_block[key]
                    #if islike and key[1] == 'data_vector':
                    #    print(module,todo.pipeline.pipe_block.get(*key,[]))
            todo()
            if islike:
                for key in join:
                    if key in module.pipe_block:
                        join[key] = cls._join_values(key,join[key],module.pipe_block[key])


    def setup(self):
        self.pipe_block = self.data_block.copy()
        join = {(section_names.data,'data_vector'):None,(section_names.model,'collection'):None,(section_names.data,'y'):None}
        self._run_todos(self.setup_todos,join)
        self.set_data()
        self.set_covariance()

    def execute(self):
        self.pipe_block = self.data_block.copy()
        join = {(section_names.model,'collection'):None,(section_names.model,'y'):None}
        self._run_todos(self.execute_todos,join)
        for key,value in join.items():
            if value is not None:
                self.pipe_block[key] = value
        self.set_model()
        self.data_block[section_names.likelihood,'loglkl'] = self.loglkl()
