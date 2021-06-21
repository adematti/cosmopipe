import re

import numpy as np

from cosmopipe.lib.utils import BaseClass


class ProjectionName(BaseClass):

    MULTIPOLE = 'multipole'
    MUWEDGE = 'muwedge'
    PIWEDGE = 'piwedge'
    _shorts = {MULTIPOLE:'ell',MUWEDGE:'mu',PIWEDGE:'pi'}
    _latex = {MULTIPOLE:'\ell',MUWEDGE:'\mu',PIWEDGE:'\pi'}

    def __init__(self, args):
        if isinstance(args,self.__class__):
            self.__dict__.update(args.__dict__)
            return
        if isinstance(args,str):
            for key,short in self._shorts.items():
                match = re.match('{}_(.*)$'.format(short),args)
                if match:
                    break
            if key == 'multipole':
                args = (key,int(match.group(1)))
            if key.endswith('wedge'):
                tu = match.group(1).split('_')
                args = (key,tuple(eval(t,{},{}) for t in tu))
        self.mode,self.proj = args

    @property
    def latex(self):
        base = self._latex[self.mode]
        if self.mode == self.MULTIPOLE:
            return '{} = {}'.format(base,self.proj)
        return '{} = ({:.2f},{:.2f})'.format(base,*self.proj)

    def __repr__(self):
        return '{}({}_{})'.format(self.__class__.__name__,self._shorts[self.mode],self.proj)

    def __str__(self):
        if self.mode == self.MULTIPOLE:
            proj = self.proj
        else:
            proj = '_'.join([str(p) for p in self.proj])
        return '{}_{}'.format(self._shorts[self.mode],proj)

    def __eq__(self, other):
        return self.mode == other.mode and self.proj == other.proj

    def __hash__(self):
        return hash(str(self))

    def __gt__(self, other):
        return np.mean(self.proj) > np.mean(other.proj)

    def __lt__(self, other):
        return np.mean(self.proj) < np.mean(other.proj)

    def __getstate__(self):
        return {'mode':self.mode,'proj':self.proj}

    def __setstate__(self, state):
        self.mode = state['mode']
        self.proj = state['proj']
