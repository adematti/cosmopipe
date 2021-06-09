import re

import numpy as np

from cosmopipe.lib.utils import BaseClass
from .integration import MultipolesIntegration
from . import utils

from cosmopipe.lib.data import DataVector, ProjectionName


class DataVectorProjection(BaseClass):

    def __init__(self, xdata, projdata=None, basemodel='xmu', integration=None):
        self.basemodel = basemodel
        if isinstance(xdata, DataVector):
            self.x = [xdata.get_x(proj=proj) for proj in xdata.projs]
            self.projnames = xdata.projs
        else:
            self.x = xdata
            self.projnames = projdata
            if np.isscalar(self.x[0]):
                self.x = [self.x]*len(self.projnames)
            elif len(self.x) != len(self.projnames):
                raise ValueError('x and proj shapes cannot be matched.')
        self.projnames = [ProjectionName(projname) for projname in self.projnames]

        if integration is None:
            integration = {projname:None for projname in self.projnames}

        self.projections = {}
        for x,projname in zip(self.x,self.projnames):
            self.set_data_projection(x,projname,integration=integration[projname])

        self.evalmesh = [[mesh.copy() for mesh in self.projections[self.projnames[0]].evalmesh]]
        for projname,proj in self.projections.items():
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

        for projname,proj in self.projections.items():
            cmesh = self.evalmesh[proj.indexmesh][0]
            proj.maskmesh = None
            if not np.all(cmesh == proj.evalmesh[0]):
                proj.maskmesh = utils.match1d(proj.evalmesh[0],cmesh)[0]
                assert len(proj.maskmesh) == len(proj.evalmesh[0])

    def set_data_projection(self, x, projname, integration=None):
        integration = integration or {}
        #proj = ProjectionName(projname)
        if projname.mode == 'multipole':
            if self.basemodel == 'xmu':
                self.projections[projname] = MultipolesIntegration({**integration,'ells':(projname.proj,)})
                self.projections[projname].evalmesh = [x,self.projections[projname].mu]

    def __call__(self, fun, concatenate=True, **kwargs):
        evals = [fun(*mesh,**kwargs) for mesh in self.evalmesh]
        toret = []
        for projname in self.projnames:
            proj = self.projections[projname]
            mesh = evals[proj.indexmesh]
            projected = proj(mesh[proj.maskmesh,...] if proj.maskmesh is not None else mesh)
            toret.append(projected.flat)
        if concatenate:
            return np.concatenate(toret)
        return toret
