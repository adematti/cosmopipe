import logging
from collections import UserList
import math
import re

import numpy as np
from scipy import stats
from pypescript.syntax import Decoder

from . import utils
from .utils import BaseClass
from cosmopipe import section_names
from cosmopipe.lib import mpi


class ParamError(Exception):

    pass


class ParamBlock(BaseClass):

    logger = logging.getLogger('ParamBlock')

    def __init__(self, data=None, string=None, parser=None):
        if isinstance(data,ParamBlock):
            self.__dict__.update(data.__dict__)
            return

        self.data = []
        if isinstance(data,(list,tuple)):
            data_ = data
            data = {}
            for name in data_:
                if isinstance(name,Parameter):
                    data[name.name] = name
                elif isinstance(name,dict):
                    data[name['name']] = name
                else:
                    data[name] = {}

        elif not isinstance(data,dict):
            data = Decoder(data=data,string=string,parser=parser)

        for name,conf in data.items():
            if isinstance(conf,Parameter):
                self.set(conf)
            else:
                conf = Parameter(name=name,**conf)
                self.set(conf)

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

    def __copy__(self):
        new = super(ParamBlock,self).__copy__()
        new.data = []
        for param in self:
            new.set(param)
        return new


class ParamName(BaseClass):

    sep = '.'

    def __init__(self, *names):
        if len(names) == 1:
            if isinstance(names[0],Parameter):
                self.__dict__.update(names[0].name.__dict__)
                return
            if isinstance(names[0],self.__class__):
                self.__dict__.update(names[0].__dict__)
                return
            if isinstance(names[0],str):
                names = tuple(names[0].split(self.sep))
            if isinstance(names[0],(tuple,list)):
                names = tuple(names[0])
        self.tuple = tuple(str(name) for name in names)

    def add_suffix(self, suffix):
        self.tuple = self.tuple[::-1] + ('{}_{}'.format(self.tuple[-1],suffix),)

    def __repr__(self):
        return '{}{}'.format(self.__class__.__name__,self.tuple)

    def __str__(self):
        return self.sep.join(self.tuple)

    def split(self):
        return str(self).split(self.sep)

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
            setattr(self,key,Prior.from_state(state[key]))

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__,self.name,'fixed' if self.fixed else 'varied')

    def __str__(self):
        return str(self.name)

    def __eq__(self, other):
        return type(other) == type(self) and all(getattr(other,key) == getattr(self,key) for key in self._keys)


class PriorError(Exception):

    pass


class Prior(BaseClass):

    logger = logging.getLogger('Prior')

    def __init__(self, dist='uniform', limits=None, **kwargs):

        if isinstance(dist,Prior):
            self.__dict__.update(dist.__dict__)
            return

        self.set_limits(limits)
        self.dist = dist
        self.attrs = kwargs

        # improper prior
        if not self.is_proper():
            return

        if self.is_limited():
            dist = getattr(stats,self.dist if self.dist.startswith('trunc') or self.dist == 'uniform' else 'trunc{}'.format(self.dist))
            if self.dist == 'uniform':
                self.rv = dist(self.limits[0],self.limits[1]-self.limits[0])
            else:
                self.rv = dist(*self.limits,**kwargs)
        else:
            self.rv = getattr(scipy.stats,self.dist)(**kwargs)

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
        return self.limits[0] < x < self.limits[1]

    def __call__(self, x):
        if not self.is_proper():
            return 1.
        return self.logpdf(x)

    def sample(self, size=None, random_state=None):
        if not self.is_proper():
            raise PriorError('Cannot sample from improper prior')
        return self.rvs(size=size,random_state=random_state)

    def __str__(self):
        base = self.dist
        if self.is_limited():
            base = '{}[{}, {}]'.format(self.dist,*self.limits)
        return '{}({})'.format(base,self.attrs)

    def __setstate__(self, state):
        self.__init__(state['dist'],state['limits'],**state['attrs'])

    def __getstate__(self):
        state = {}
        for key in ['dist','limits','attrs']:
            state[key] = getattr(self,key)
        return state

    def is_proper(self):
        return self.dist != 'uniform' or not np.isinf(self.limits).any()

    def is_limited(self):
        return not np.isinf(self.limits).all()

    def __getattribute__(self, name):
        try:
            return object.__getattribute__(self,name)
        except AttributeError:
            attrs = object.__getattribute__(self,'attrs')
            if name in attrs:
                return attrs[name]
            rv = object.__getattribute__(self,'rv')
            return getattr(rv,name)

    def __eq__(self, other):
        return type(other) == type(self) and all(getattr(other,key) == getattr(self,key) for key in ['dist','limits','attrs'])
