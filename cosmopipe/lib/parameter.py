import logging
from collections import UserList
import math
import re

import numpy as np
from pypescript.config import ConfigBlock

from . import utils
from .utils import BaseClass
from cosmopipe import section_names
from cosmopipe.lib import mpi


class ParamError(Exception):

    pass


class ParamBlock(BaseClass):

    logger = logging.getLogger('ParamBlock')

    def __init__(self, filename=None, string=None, parser=utils.parse_yaml):
        data = {}
        self.data = []
        if isinstance(filename,self.__class__):
            self.update(filename)
            return
        elif isinstance(filename,str):
            self.filename = filename
            with open(filename,'r') as file:
                if string is None: string = ''
                string += file.read()
        elif isinstance(filename,(list,tuple)):
            for name in filename:
                if isinstance(name,Parameter):
                    data[name.name] = name
                elif isinstance(name,dict):
                    data[name['name']] = name
                else:
                    data[name] = {}
        elif filename is not None:
            data = dict(filename)

        if string is not None and parser is not None:
            data.update(parser(string))

        for name,conf in data.items():
            if isinstance(conf,Parameter):
                self.set(conf)
            elif any(key in Parameter._keys for key in conf.keys()) or not conf:
                conf = Parameter(name=name.split('.'),**conf)
                self.set(conf)
            else:
                for name2,conf2 in conf.items():
                    if not isinstance(conf2,Parameter):
                         conf2 = Parameter(name=(name,name2),**conf2)
                    self.set(conf2)

    def __getitem__(self, name):
        if isinstance(name,Parameter):
            if name not in self.data:
                raise KeyError('Parameter {} not found'.format(name.name))
            return self.data[self.data.index(name)]
        try:
            return self.data[name]
        except TypeError:
            return self.data[self._index_name(name)]

    def __setitem__(self, name, item):
        if not isinstance(item,Parameter):
            raise TypeError('{} is not a Parameter instance.'.format(item))
        if isinstance(name,Parameter):
            if name not in self.data:
                raise KeyError('Parameter {} not found'.format(name.name))
            self.data[self.data.index(name)] = item
            return
        try:
            self.data[name] = item
        except TypeError:
            name = ParamName(name)
            if item.name != name:
                raise KeyError('Parameter {} should be indexed by name (incorrect {})'.format(item.name,name))
            self.data[self._index_name(name)] = item

    def __delitem__(self, name):
        try:
            del self[name]
        except TypeError:
            del self[self._index_name(name)]

    def set(self, param):
        if param in self:
            self[param.name] = param
        else:
            self.data.append(param)

    def names(self):
        return (item.name for item in self.data)

    def index(self, name):
        if isinstance(name,Parameter):
            return self.data.index(name)
        name = ParamName(name)
        return self._index_name(name)

    def _index_name(self, name):
        return list(self.names()).index(name)

    def __contains__(self, name):
        if not isinstance(name,Parameter):
            return name in self.names()
        return name in self.data

    def update(self, other):
        for param in other:
            self.set(param)

    def setdefault(self, param):
        if param.name not in self:
            self.set(param)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__,self.data)

    def __len__(self):
        return len(self.data)

    def __eq__(self, other):
        return other.data == self.data

    def __getstate__(self):
        return {'data':[param.__getstate__() for param in self]}

    def __setstate__(self, state):
        self.data = [Parameter.from_state(param) for param in state['data']]

    def select(self, **kwargs):
        toret = self.__class__()
        for param in self:
            if all(getattr(param,key) == val for key,val in kwargs.items()):
                toret.set(param)
        return toret


class ParamName(BaseClass):

    def __init__(self, *names):
        if len(names) == 1:
            if isinstance(names[0],Parameter):
                self.__dict__.update(names[0].name.__dict__)
                return
            if isinstance(names[0],self.__class__):
                self.__dict__.update(names[0].__dict__)
                return
            if isinstance(names[0],str):
                names = tuple(names[0].split('.'))
            if isinstance(names[0],(tuple,list)):
                names = tuple(names[0])
        self.tuple = tuple(str(name) for name in names)

    def add_suffix(self, suffix):
        self.tuple = self.tuple[::-1] + ('{}_{}'.format(self.tuple[-1],suffix),)

    def __repr__(self):
        return '{}{}'.format(self.__class__.__name__,self.tuple)

    def __str__(self):
        return '.'.join(self.tuple)

    def split(self, sep='.'):
        return str(self).split(sep)

    def __eq__(self, other):
        if isinstance(other,str):
            return other == str(self)
        if isinstance(other,tuple):
            return other == self.tuple
        return isinstance(other,self.__class__) and other.tuple == self.tuple

    def __hash__(self):
        return hash(str(self))

    def __getstate__(self):
        return {'data':self.tuple}

    def __setstate__(self, state):
        self.tuple = state['data']


class Parameter(BaseClass):

    _keys = ['name','value','latex','fixed','proposal','prior','ref']
    logger = logging.getLogger('Parameter')

    def __init__(self, name=None, value=None, fixed=None, prior=None, ref=None, proposal=None, latex=None):
        if isinstance(name,Parameter):
            self.__dict__.update(name.__dict__)
            return
        self.name = ParamName(name)
        self.value = value
        self.prior = Prior(**(prior or {}))
        if value is None:
            if self.prior.is_proper():
                self.value = np.mean(self.prior.limits)
        if ref is not None:
            self.ref = Prior(**ref)
        else:
            self.ref = self.prior.copy()
        if value is None:
            if (ref is not None or prior is not None):
                if hasattr(self.ref,'loc'):
                    self.value = self.ref.loc
                elif self.ref.is_proper():
                    self.value = (self.ref.limits[1] - self.ref.limits[0])/2.
        self.latex = latex
        if fixed is None:
            fixed = prior is None and ref is None
        self.fixed = bool(fixed)
        self.proposal = proposal
        if proposal is None:
            if (ref is not None or prior is not None):
                if hasattr(self.ref,'scale'):
                    self.proposal = self.ref.scale
                elif self.ref.is_proper():
                    self.proposal = (self.ref.limits[1] - self.ref.limits[0])/2.

    def add_suffix(self, suffix):
        self.name.add_suffix(suffix)
        if self.latex is not None:
            match1 = re.match('(.*)_(.)$',self.latex)
            match2 = re.match('(.*)_{(.*)}$',self.latex)
            if match1 is not None:
                self.latex = '%s_{%s,\\mathrm{%s}}' % (match1.group(1),match1.group(2),self.name)
            elif match2 is not None:
                self.latex = '%s_{%s,\\mathrm{%s}}' % (match2.group(1),match2.group(2),self.name)
            else:
                self.latex = '%s_{\\mathrm{%s}}' % (self.latex,self.name)

    def get_label(self):
        if self.latex is not None:
            return '${}$'.format(self.latex)
        return self.name

    @property
    def limits(self):
        return self.prior.limits

    def __getstate__(self):
        state = {}
        for key in self._keys:
            state[key] = getattr(self,key)
            if hasattr(state[key],'__getstate__'):
                state[key] = state[key].__getstate__()
        return state

    def __setstate__(self, state):
        super(Parameter,self).__setstate__(state)
        self.name = ParamName.from_state(state['name'])
        for key in ['prior','ref']:
            setattr(self,key,Prior(**state[key]))

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__,self.name,'fixed' if self.fixed else 'varied')

    def __str__(self):
        return str(self.name)

    def __eq__(self, other):
        return type(other) == type(self) and all(getattr(other,key) == getattr(self,key) for key in self._keys)


def Prior(dist='uniform', limits=None, **kwargs):

    if isinstance(dist,BasePrior):
        dist = dist.copy()
        if limits is not None:
            dist.set_limits(limits)
        return dist

    if dist.lower() in BasePrior.registry:
        cls = BasePrior.registry[dist.lower()]
    else:
        raise ParamError('Unable to understand prior {}; it should be one of {}'.format(dist,list(prior_registry.keys())))

    return cls(**kwargs, limits=limits)


class PriorError(Exception):

    pass


class BasePrior(BaseClass):

    logger = logging.getLogger('BasePrior')
    _keys = []

    def set_limits(self, limits=None):
        if not limits:
            limits = (-np.inf,np.inf)
        self.limits = list(limits)
        if self.limits[0] is None: self.limits[0] = -np.inf
        if self.limits[1] is None: self.limits[1] = np.inf
        self.limits = tuple(self.limits)
        if self.limits[1] <= self.limits[0]:
            raise PriorError('Prior range {} has min greater than max'.format(self.limits))
        if np.isinf(self.limits).any():
            return 1
        return 0

    def isin(self, x):
        return  self.limits[0] < x < self.limits[1]

    def __call__(self, x):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

    def __setstate__(self,state):
        super(BasePrior,self).__setstate__(state)
        self.set_limits(self.limits)

    def __getstate__(self):
        state = {}
        for key in ['dist','limits'] + self._keys:
            state[key] = getattr(self,key)
        return state

    def is_limited(self):
        return not np.isinf(self.limits).all()

    def is_proper(self):
        return True

    def __eq__(self, other):
        return type(other) == type(self) and all(getattr(other,key) == getattr(self,key) for key in ['limits'] + self._keys)


class UniformPrior(BasePrior):

    logger = logging.getLogger('UniformPrior')

    def __init__(self, limits=None):
        self.set_limits(limits)

    def set_limits(self, limits=None):
        if super(UniformPrior,self).set_limits(limits) == 1:
            self.norm = 0. # we tolerate improper priors
        else:
            self.norm = -np.log(limits[1] - limits[0])

    def __call__(self, x, norm=True):
        if not self.isin(x):
            return -np.inf
        return self.norm if norm else 0

    def __str__(self):
        return '{}({}, {})'.format(self.dist,*self.limits)

    def sample(self, size=None, seed=None, rng=None):
        if not self.is_proper():
            raise PriorError('Cannot sample from improper prior')
        self.rng = rng or np.random.RandomState(seed=seed)
        return self.rng.uniform(*self.limits,size=size)

    def is_proper(self):
        return not np.isinf(self.limits).any()


class NormPrior(BasePrior):

    logger = logging.getLogger('NormPrior')
    _keys = ['loc','scale']

    def __init__(self, loc=0., scale=1., limits=None):
        self.loc = loc
        self.scale = scale
        self.set_limits(limits)

    @property
    def scale2(self):
        return self.scale**2

    def set_limits(self, limits):
        super(NormPrior,self).set_limits(limits)

        def cdf(x):
            return 0.5*(math.erf(x/math.sqrt(2.)) + 1)

        a,b = [(x-self.loc)/self.scale for x in self.limits]
        self.norm = np.log(cdf(b) - cdf(a)) + 0.5*np.log(2*np.pi*self.scale**2)

    def __call__(self, x, norm=True):
        if not self.isin(x):
            return -np.inf
        return -0.5 * ((x-self.loc) / self.scale)**2 - (self.norm if norm else 0.)

    def __str__(self):
        return '{}({}, {})'.format(self.dist,self.loc,self.scale)

    def sample(self, size=None, seed=None, rng=None):
        self.rng = rng or np.random.RandomState(seed=seed)
        if self.limits == (-np.inf,np.inf):
            return self.rng.normal(loc=self.loc,scale=self.scale,size=size)
        samples = []
        isscalar = size is None
        if isscalar: size = 1
        while len(samples) < size:
            x = self.rng.normal(loc=self.loc,scale=self.scale)
            if self.isin(x):
                samples.append(x)
        if isscalar:
            return samples[0]
        return np.array(samples)


BasePrior.registry = {}
for cls in BasePrior.__subclasses__():
    dist = cls.__name__[:-len('Prior')].lower()
    cls.dist = dist
    BasePrior.registry[dist] = cls
