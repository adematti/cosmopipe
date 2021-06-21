import re

import numpy as np

from cosmopipe.lib.utils import BaseClass
from .integration import MultipoleIntegration, MuWedgeIntegration, MultipoleToMultipole, MultipoleToMuWedge
from . import utils

from cosmopipe.lib.data import DataVector, ProjectionName


class ProjectionBase(BaseClass):

    MULTIPOLE = 'multipole'
    MUWEDGE = 'muwedge'

    def __init__(self, *args):
        if isinstance(args[0],self.__class__):
            self.__dict__.update(args[0].__dict__)
            return
        if len(args) == 1:
            if np.isscalar(args[0]):
                self.mode,self.projs = args[0],None
            else:
                self.mode,self.projs = args[0]
        else:
            self.mode,self.projs = args

    def __repr__(self):
        return '{}({},{})'.format(self.__class__.__name__,self.mode,self.projs)

    def __getstate__(self):
        return {'mode':self.mode,'projs':self.projs}

    def __setstate__(self, state):
        self.mode = state['mode']
        self.projs = state['projs']


class DataVectorProjection(BaseClass):

    def __init__(self, x, projs=None, model_base='muwedge', integration=None):
        self.model_base = ProjectionBase(model_base)
        if projs is None:
            projs = x.get_projs()
        self.projs = [ProjectionName(projname) for projname in projs]
        if isinstance(x,DataVector):
            self.x = [x.get_x(proj=proj) for proj in projs]
        else:
            self.x = x
            if np.ndim(self.x[0]) == 0:
                self.x = [self.x]*len(self.projs)
            elif len(self.x) != len(self.projs):
                raise ValueError('x and proj shapes cannot be matched.')

        if integration is None:
            integration = {projname:None for projname in self.projs}
        self.integration_options = integration

        self.integration_engines = {}
        for x,projname in zip(self.x,self.projs):
            self.set_data_projection(x,projname,integration=self.integration_options[projname])

        self.evalmesh = [[mesh.copy() for mesh in self.integration_engines[self.projs[0]].evalmesh]]
        for projname,proj in self.integration_engines.items():
            proj.indexmesh = None
            for imesh,mesh in enumerate(self.evalmesh):
                if all(np.all(m1 == m2) for m1,m2 in zip(mesh[1:],proj.evalmesh[1:])):
                    mesh[0] = np.concatenate([mesh[0],proj.evalmesh[0]])
                    proj.indexmesh = imesh
            if proj.indexmesh is None:
                proj.indexmesh = len(self.evalmesh)
                self.evalmesh.append(proj.evalmesh)

        for mesh in self.evalmesh:
            uniques,indices = np.unique(mesh[0],return_index=True)
            if len(indices) < len(mesh[0]):
                # to preserve initial order
                mask = np.zeros(len(mesh[0]),dtype='?')
                mask[indices] = True
                mesh[0] = mesh[0][mask]

        for projname,proj in self.integration_engines.items():
            cmesh = self.evalmesh[proj.indexmesh][0]
            proj.maskmesh = None
            if not np.all(cmesh == proj.evalmesh[0]):
                proj.maskmesh = utils.match1d(proj.evalmesh[0],cmesh)[0]
                assert len(proj.maskmesh) == len(proj.evalmesh[0])

    def set_data_projection(self, x, projname, integration=None):
        integration = integration or {}
        #proj = ProjectionName(projname)
        error = ValueError('Unknown projection {} -> {}'.format(self.model_base.mode,projname.mode))
        if projname.mode == ProjectionName.MULTIPOLE:
            if self.model_base.mode == ProjectionBase.MUWEDGE:
                self.integration_engines[projname] = MultipoleIntegration({**integration,'ells':(projname.proj,)})
                self.integration_engines[projname].evalmesh = [x,self.integration_engines[projname].mu]
            elif self.model_base.mode == ProjectionBase.MULTIPOLE:
                self.integration_engines[projname] = MultipoleToMultipole(ellsin=self.model_base.projs,ells=(projname.proj,))
                self.integration_engines[projname].evalmesh = [x]
            else:
                raise error
        elif projname.mode == ProjectionName.MUWEDGE:
            if self.model_base.mode == ProjectionBase.MUWEDGE:
                self.integration_engines[projname] = MuWedgeIntegration({**integration,'muwedges':(projname.proj,)})
                self.integration_engines[projname].evalmesh = [x,self.integration_engines[projname].mu[0]]
            elif self.model_base.mode == ProjectionBase.MULTIPOLE:
                self.integration_engines[projname] = MultipoleToMuWedge(ellsin=self.model_base.projs,muwedges=(projname.proj,))
                self.integration_engines[projname].evalmesh = [x]
            else:
                raise error
        else:
            raise error

    def __call__(self, fun, concatenate=True, **kwargs):
        evals = [fun(*mesh,**kwargs) for mesh in self.evalmesh]
        toret = []
        for projname in self.projs:
            proj = self.integration_engines[projname]
            mesh = evals[proj.indexmesh]
            projected = proj(mesh[proj.maskmesh,...] if proj.maskmesh is not None else mesh)
            toret.append(projected.flatten())
        if concatenate:
            return np.concatenate(toret)
        return toret

    def to_data_vector(self, fun, **kwargs):
        from cosmopipe.lib.data import DataVector
        y = self(fun,concatenate=False,**kwargs)
        return DataVector(x=self.x,y=y,mapping_proj=self.projs)
