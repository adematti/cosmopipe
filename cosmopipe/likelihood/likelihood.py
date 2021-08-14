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
        self.data_block[section_names.parameters,'list'] = self.pipe_block[section_names.parameters,'list']
        #self.data_block[section_names.likelihood,'likelihood'] = self

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
        self.data_block[section_names.likelihood,'loglkl'] = self.pipe_block[section_names.likelihood,'loglkl'] = self.loglkl()


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


class SumLikelihood(BasePipeline):

    logger = logging.getLogger('SumLikelihood')

    def __init__(self, *args, **kwargs):
        super(SumLikelihood,self).__init__(*args,**kwargs)
        like = self.options.get_list('like',None)
        self.like_modules = []
        if like is None:
            for module in self.modules.values():
                if isinstance(module,BaseLikelihood):
                    self.log_info('Found likelihood {}.'.format(module.name))
                    self.like_modules.append(module)
        else:
            from pypescript.pipeline import ModuleTodo, syntax
            for module in like:
                module = self.add_module(module)
                self.like_modules.append(module)
                self.setup_todos.append(ModuleTodo(self,module,step=syntax.setup_function))
                self.execute_todos.append(ModuleTodo(self,module,step=syntax.execute_function))
                self.cleanup_todos.append(ModuleTodo(self,module,step=syntax.cleanup_function))
                self.config_block.update(module.config_block)
        if not self.like_modules:
            raise ValueError('No likelihood found')

    @classmethod
    def _join_values(cls, key, values):
        if isinstance(values[0],np.ndarray):
            return np.concatenate(values,axis=0)
        return sum(values)

    def _run_todos(self, todos, *join):
        self.pipe_block = self.data_block.copy()
        djoin = {key:[] for key in join}
        for todo in todos:
            todo()
            if todo.module in self.like_modules:
                for key in djoin: djoin[key].append(todo.module.pipe_block[key])
        for key,values in djoin.items():
            self.pipe_block[key] = self._join_values(key,values)

    def setup(self):
        self._run_todos(self.setup_todos,(section_names.parameters,'list'))
        self.data_block[section_names.parameters,'list'] = self.pipe_block[section_names.parameters,'list']

    def execute(self):
        self._run_todos(self.execute_todos,(section_names.likelihood,'loglkl'))
        self.data_block[section_names.likelihood,'loglkl'] = self.pipe_block[section_names.likelihood,'loglkl']


class JointGaussianLikelihood(SumLikelihood,GaussianLikelihood):

    logger = logging.getLogger('JointGaussianLikelihood')

    def __init__(self, *args, **kwargs):
        super(JointGaussianLikelihood,self).__init__(*args,**kwargs)
        after = self.options.get_list('after',[])
        self.after_setup_todos = []
        from pypescript.pipeline import ModuleTodo, syntax
        self.after_modules = []
        for module in self.after_modules:
            module = self.add_module(module)
            self.after_modules.append(module)
            self.after_setup_todos.append(ModuleTodo(self,module,step=syntax.execute_function))
            self.cleanup_todos.append(ModuleTodo(self,module,step=syntax.cleanup_function))

    def setup(self):
        #join = {(section_names.data,'data_vector'):[],(section_names.data,'y'):[],(section_names.model,'collection'):[]}
        join = [(section_names.parameters,'list'),(section_names.data,'y')]
        if self.after_setup_todos:
            join += [(section_names.data,'data_vector'),(section_names.model,'collection')]
        self._run_todos(self.setup_todos,*join)
        self.data_block[section_names.parameters,'list'] = self.pipe_block[section_names.parameters,'list']
        for todo in self.after_setup_todos:
            todo()
        self.set_data()
        self.set_covariance()

    def execute(self):
        join = [(section_names.model,'y')]
        self._run_todos(self.execute_todos,*join)
        self.set_model()
        self.data_block[section_names.likelihood,'loglkl'] = self.loglkl()
