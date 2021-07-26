import re

import numpy as np

from cosmopipe.lib.utils import BaseNameSpace, BaseOrderedCollection


class ProjectionName(BaseNameSpace):

    MULTIPOLE = 'multipole'
    MUWEDGE = 'muwedge'
    MUBIN = 'mubin'
    PIWEDGE = 'piwedge'
    CORRELATION = 'correlation'
    POWER = 'power'
    _mode_shorts = {MULTIPOLE:'ell',MUWEDGE:'mu',MUBIN:'mubin',PIWEDGE:'pi',None:'None'}
    _space_shorts = {POWER:'power',CORRELATION:'corr'}
    _latex = {MULTIPOLE:'\ell',MUWEDGE:'\mu',MUBIN:'\mu',PIWEDGE:'\pi'}
    _attrs = ['name','fields','space','mode','proj','wa_order']

    def __init__(self, *args, **kwargs):
        for name in self._attrs:
            setattr(self,name,None)
        if not len(args):
            pass
        elif len(args) > 1:
            if len(args) == 2:
                self.mode,self.proj = args
            else:
                self.name,self.mode,self.projs = args
        elif isinstance(args[0],self.__class__):
            self.__dict__.update(args[0].__dict__)
        elif isinstance(args[0],dict):
            kwargs = args[0]
        elif isinstance(args[0],(list,tuple)):
            self.__init__(*args[0],**kwargs)
        elif isinstance(args[0],str):
            args = args[0].split('_')
            self.name, self.space, self.mode = None, None, None
            for name,short in self._space_shorts.items():
                if args[0] == short:
                    self.space = name
                    args = args[1:]
                    break
                if args[1] == short:
                    self.name = args[0]
                    self.space = name
                    args = args[2:]
                    break
            for name,short in self._mode_shorts.items():
                if args[0] == short:
                    self.mode = name
                    args = args[1:]
                    break
                if self.name is None and args[1] == short:
                    self.name = args[0]
                    self.mode = name
                    args = args[2:]
                    break
            self.proj = tuple(eval(t,{},{}) for t in args)
            if len(args) == 1:
                self.proj = self.proj[0]
            if self.mode is None:
                raise ValueError('Cannot read projection {}'.format(args))
        self.set(**kwargs)

    def set(self, **kwargs):
        for name,value in kwargs.items():
            setattr(self,name,value)
        if np.ndim(self.proj):
            self.proj = tuple(self.proj)

    @property
    def latex(self):
        base = self._latex[self.mode]
        isscalar = np.ndim(self.proj) == 0
        proj = (self.proj,) if isscalar else self.proj
        label = ','.join(['{}'.format(p) if self.mode == self.MULTIPOLE else '{:.2f}'.format(p) for p in proj if p is not None])
        if not isscalar:
            label = '({})'.format(label)
        return '{} = {}'.format(base,label)

    def get_projlabel(self):
        if self.mode is None:
            return None
        return '${}$'.format(self.latex)

    def get_xlabel(self):
        if self.space == self.POWER:
            return '$k$ [$h \ \\mathrm{Mpc}^{-1}$]'
        if self.space == self.CORRELATION:
            return '$s$ [$\\mathrm{Mpc} / h$]'

    def get_ylabel(self):
        if self.space == self.POWER:
            return '$P(k)$ [$(\\mathrm{Mpc} \ h)^{-1})^{3}$]'
        if self.space == self.CORRELATION:
            return '$\\xi(s)$'

    """
    def __str__(self):
        proj = (self.proj,) if np.ndim(self.proj) == 0 else self.proj
        proj = '_'.join([str(p) for p in proj])
        tmp = '{}_{}'.format(self._mode_shorts[self.mode],proj)
        if self.name is not None:
            tmp = '{}_{}'.format(self.name,tmp)
        return tmp
    """
    def __gt__(self, other):
        return np.mean(self.proj) > np.mean(other.proj)

    def __lt__(self, other):
        return np.mean(self.proj) < np.mean(other.proj)


class ProjectionNameCollection(BaseOrderedCollection):

    _cast = lambda x: x if isinstance(x,ProjectionName) else ProjectionName(x)

    def index(self, proj, ignore_none=False):
        proj = self.__class__._cast(proj)
        if ignore_none:
            return [iproj_ for iproj_,proj_ in enumerate(self.data) if proj.eq_ignore_none(proj_)]
        if proj not in self.data:
            raise KeyError('Projection {} not found among {}'.format(proj,self.data))
        return self.data.index(proj)

    def get(self, proj, ignore_none=True):
        if ignore_none:
            return [self.data[ii] for ii in self.index(proj,ignore_none=ignore_none)]
        return self.data[self.data.index(proj)]

    def group_by(self, include=None, exclude=None):
        if not len(self):
            return {}
        include = include or []
        exclude = exclude or []
        exclude = exclude + [key for key in ProjectionName._attrs if key not in include]
        exclude = {key:None for key in exclude}
        toret = {}
        for proj in self.data:
            base = proj.copy(**exclude)
            if base not in toret:
                toret[base] = []
            toret[base].append(proj)
        return toret
