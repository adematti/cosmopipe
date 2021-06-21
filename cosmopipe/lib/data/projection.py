import re

import numpy as np

from cosmopipe.lib.utils import BaseClass


def find_in_dict(di, item):
    for key,value in di.items():
        if value == item:
            return key
    raise ValueError('Unknown {}'.format(item))


class ProjectionName(BaseClass):

    MULTIPOLE = 'multipole'
    MUWEDGE = 'muwedge'
    PIWEDGE = 'piwedge'
    CORRELATION = 'correlation'
    POWER = 'power'
    _mode_shorts = {MULTIPOLE:'ell',MUWEDGE:'mu',PIWEDGE:'pi'}
    _space_shorts = {POWER:'power',CORRELATION:'corr'}
    _latex = {MULTIPOLE:'\ell',MUWEDGE:'\mu',PIWEDGE:'\pi'}

    def __init__(self, args):
        if isinstance(args,self.__class__):
            self.__dict__.update(args.__dict__)
            return
        if args is None:
            self.space,self.mode,self.proj = None,None,None
            return
        if isinstance(args,str):
            space = None
            for name,short in self._space_shorts.items():
                match = re.match('{}_(.*)$'.format(short),args)
                if match:
                    space = name
                    args = match.group(1)
                    break
            mode = None
            for name,short in self._mode_shorts.items():
                match = re.match('{}_(.*)$'.format(short),args)
                if match:
                    mode = name
                    proj = match.group(1)
                    if mode == 'multipole':
                        proj = int(proj)
                    if mode.endswith('wedge'):
                        tu = proj.split('_')
                        proj = tuple(eval(t,{},{}) for t in tu)
                    break
            if mode is None:
                raise ValueError('Cannot read projection {}'.format(args))
            args = (space,mode,proj)
        if len(args) == 2:
            self.space = None
            self.mode,self.proj = args
        else:
            self.space,self.mode,self.proj = args

    @property
    def latex(self):
        base = self._latex[self.mode]
        if self.mode == self.MULTIPOLE:
            return '{} = {}'.format(base,self.proj)
        return '{} = ({:.2f},{:.2f})'.format(base,*self.proj)

    def get_projlabel(self):
        if self.mode is None:
            return None
        return '${}$'.format(self.latex)

    def get_xlabel(self):
        if self.space == self.POWER:
            return '$k$ [$h \ \\mathrm{Mpc}^{-1}$]'
        if self.space == self.CORRELATION:
            return '$s$ [$\\mathrm{Mpc} \ h$]'

    def get_ylabel(self):
        if self.space == self.POWER:
            return '$P(k)$ [$(\\mathrm{Mpc} \ h)^{-1})^{3}$]'
        if self.space == self.CORRELATION:
            return '$\\xi(s)$'

    def __repr__(self):
        return '{}({}_{}_{})'.format(self.__class__.__name__,self._space_shorts[self.space],self._mode_shorts[self.mode],self.proj)

    def __str__(self):
        if self.mode == self.MULTIPOLE:
            proj = self.proj
        else:
            proj = '_'.join([str(p) for p in self.proj])
        if self.space is None:
            return '{}_{}'.format(self._mode_shorts[self.mode],proj)
        if self.mode is None:
            return self._space_shorts[self.space]
        return '{}_{}_{}'.format(self._space_shorts[self.space],self._mode_shorts[self.mode],proj)

    def __eq__(self, other):
        return (self.space == other.space or self.space is None or other.space is None) and self.mode == other.mode and self.proj == other.proj

    def __hash__(self):
        return hash(str(self))

    def __gt__(self, other):
        return np.mean(self.proj) > np.mean(other.proj)

    def __lt__(self, other):
        return np.mean(self.proj) < np.mean(other.proj)

    def __getstate__(self):
        return {name:getattr(self,name) for name in ['space','mode','proj']}

    def __setstate__(self, state):
        for name in ['space','mode','proj']:
            setattr(self,name,state[name])
