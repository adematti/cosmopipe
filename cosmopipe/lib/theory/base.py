import functools

import numpy as np
from scipy import interpolate
from cosmoprimo import PowerSpectrumInterpolator1D

from cosmopipe.lib.utils import BaseClass, BaseNameSpace, BaseOrderedCollection
from cosmopipe.lib.data_vector import ProjectionName, ProjectionNameCollection
from .fog import get_FoG


class ProjectionBase(BaseNameSpace):

    MULTIPOLE = 'multipole'
    MUWEDGE = 'muwedge'
    POWER = 'power'
    CORRELATION = 'correlation'
    _attrs = ['x','name','fields','space','shotnoise','mode','projs','wa_order']

    def __init__(self, x=None, **kwargs):
        if isinstance(x,self.__class__):
            self.__dict__.update(x.__dict__)
            return
        for name in self._attrs:
            setattr(self,name,None)
        if isinstance(x,dict):
            self.__init__(**x)
            return
        self.x = x
        self.set(**kwargs)

    def __gt__(self, other):
        return np.mean(self.projs) > np.mean(other.projs)

    def __lt__(self, other):
        return np.mean(self.projs) < np.mean(other.projs)

    def __hash__(self):
        # if __eq__ is redefined, __hash__ must be as well
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other,self.__class__) and all(getattr(self,name) == getattr(other,name) for name in self._attrs if name != 'x')

    def to_projs(self):
        toret = ProjectionNameCollection()
        di = {name:value for name,value in self.as_dict(drop_none=True).items() if name in ProjectionName._attrs and name != 'proj'}
        return ProjectionNameCollection([ProjectionName(proj=proj,**di) for proj in self.projs])


class ProjectionBaseCollection(BaseOrderedCollection):

    _cast = ProjectionBase

    def spaces(self):
        return [base.space for data in self.data]

    def modes(self):
        return [base.mode for mode in self.data]

    def get_by_proj(self, *args, **kwargs):
        proj = ProjectionName(*args,**kwargs)
        indices = range(len(self))
        if proj.name is not None:
            indices = [index for index in indices if self.data[index].name == proj.name]
        if proj.space is not None:
            indices = [index for index in indices if self.data[index].space == proj.space]
        if len(indices) > 1 and proj.mode is not None:
            indices = [index for index in indices if self.data[index].mode == proj.mode]
        if len(indices) > 1 and proj.wa_order is not None:
            indices = [index for index in indices if self.data[index].wa_order == proj.wa_order]
        if not len(indices):
            raise IndexError('Could not find any match between data projection {} and model bases {}'.format(proj,self))
        if len(indices) > 1:
            raise IndexError('Data projection {} corresponds to several model bases {}'.format(proj,[self.data[ii] for ii in indices]))
        return self.data[indices[0]]


class ModelCollection(BaseClass):

    def __init__(self, models=None, bases=None):
        if isinstance(models,self.__class__):
            self.__dict__.update(models.__dict__)
            return
        self.data = {}
        if models is None:
            models = []
        if not isinstance(models,list):
            models = [models]
        if bases is None:
            bases = [None]*len(models)
        for model,base in zip(models,bases):
            self.set(model,base=base)

    @property
    def bases(self):
        return ProjectionBaseCollection(list(self.data.keys()))

    @property
    def models(self):
        return list(self.data.values())

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__,repr(self.data))

    def __copy__(self):
        new = super(ModelCollection,self).__copy__()
        new.data = self.data.copy()
        return new

    def get(self, base):
        base = ProjectionBase(base)
        return self.data[base]

    def set(self, model, base=None):
        if base is None:
            base = model.base
        else:
            base = ProjectionBase(base)
        self.data[base] = model

    def select(self, *args, **kwargs):
        new = self.__class__()
        bases = self.bases.select(*args,**kwargs)
        for base in bases:
            new.set(self.get(base),base=base)
        return new

    def __getitem__(self, base):
        return self.get(base)

    def __len__(self):
        return len(self.data)

    def __contains__(self, base):
        return ProjectionBase(base) in self.data

    def __iter__(self):
        return iter(self.data.items())

    def get_by_proj(self, *args, **kwargs):
        base = self.bases.get_by_proj(*args,**kwargs)
        return self.data[base]

    @classmethod
    def concatenate(cls, *others):
        new = cls(others[0])
        new.data = others[0].data.copy()
        for other in others[1:]:
            for base,model in cls(other).data.items():
                new.set(model,base=base)
        return new

    def extend(self, other):
        new = self.concatenate(self,other)
        self.__dict__.update(new.__dict__)

    def __radd__(self, other):
        if other in [[],0,None]:
            return self.copy()
        return self.concatenate(self,other)

    def __add__(self, other):
        return self.concatenate(self,other)


class BaseModel(BaseClass):

    def __init__(self, base=None):
        self.base = ProjectionBase(base or {})

    def __call__(self, *args, **kwargs):
        return self.eval(*args,**kwargs)


class BasePTModel(BaseModel):

    def __init__(self, pklin, klin=None, FoG='gaussian'):
        if callable(pklin):
            self.pk_linear = pklin
            if klin is None: klin = getattr(pklin,'k',None)
        elif klin is not None:
            self.pk_linear = PowerSpectrumInterpolator1D(k=klin,pk=pklin)
        else:
            raise ValueError('Input pklin should be a PowerSpectrumInterpolator1D instance if no k provided.')
        self.FoG = FoG
        if FoG is not None:
            self.FoG = get_FoG(FoG)
        self.base = ProjectionBase(x=klin,space=ProjectionBase.POWER,mode=ProjectionBase.MUWEDGE,wa_order=0)
