import re

import numpy as np

from cosmopipe.lib.utils import BaseClass
from cosmopipe.lib.data_vector import DataVector, ProjectionName

from .base import ProjectionBase, ProjectionBaseCollection
from .integration import MultipoleIntegration, MuWedgeIntegration, MultipoleToMultipole, MultipoleToMuWedge
from . import utils


class ModelCollectionProjection(BaseClass):

    def __init__(self, data, projs=None, model_bases=None, integration=None):

        self.model_bases = ProjectionBaseCollection(model_bases)
        self.data = data
        self.projs = [ProjectionName(proj) for proj in projs] if projs is not None else data.get_projs()

        if integration is None:
            integration = {projname:None for projname in self.projs}
        self.integration_options = integration

        model_bases = {}
        for iproj,proj in enumerate(self.projs):
            base = self.model_bases.get_by_proj(proj)
            if base not in model_bases:
                model_bases[base] = []
            model_bases[base].append(iproj)

        self.model_projections = []
        self.projection_mapping = [None]*len(self.projs)
        nprojs = 0
        for model_base,projection_indices in model_bases.items():
            projs = [self.projs[ii] for ii in projection_indices]
            #data = self.data.copy().view(new=False,proj=projs)
            #print(self.data.get_projs(),data.get_projs(),projs)
            model_projection = ModelProjection(self.data,projs=projs,model_base=model_base,integration={proj:self.integration_options[proj] for proj in projs})
            self.model_projections.append(model_projection)
            for ii,jj in enumerate(projection_indices): self.projection_mapping[jj] = nprojs + ii
            nprojs += len(projs)

    def __call__(self, models, concatenate=True):
        tmp = []
        for model_projection in self.model_projections:
            tmp += model_projection(models.get(model_projection.model_base),concatenate=False)
        toret = [tmp[ii] for ii in self.projection_mapping]
        if concatenate:
            return np.concatenate(toret)
        return toret

    def to_data_vector(self, fun, **kwargs):
        from cosmopipe.lib.data_vector import DataVector
        y = self(fun,concatenate=True,**kwargs)
        data = self.data.deepcopy()
        data.set_y(y)
        return data


class ModelProjection(BaseClass):

    def __init__(self, data, projs=None, model_base=None, integration=None):
        self.model_base = ProjectionBase(model_base or {})
        self.data = data
        self.projs = [ProjectionName(proj) for proj in projs] if projs is not None else data.get_projs()

        if integration is None:
            integration = {projname:None for projname in self.projs}
        self.integration_options = integration

        self.integration_engines = {}
        self.shotnoise = {}
        for projname in self.projs:
            self.set_model_projection(data,projname,integration=self.integration_options[projname])

        self.evalmesh = [[mesh.copy() for mesh in self.integration_engines[self.projs[0]].evalmesh]]
        for projname,integ in self.integration_engines.items():
            integ.indexmesh = None
            for imesh,mesh in enumerate(self.evalmesh):
                if all(np.all(m1 == m2) for m1,m2 in zip(mesh[1:],integ.evalmesh[1:])):
                    mesh[0] = np.concatenate([mesh[0],integ.evalmesh[0]])
                    integ.indexmesh = imesh
            if integ.indexmesh is None:
                integ.indexmesh = len(self.evalmesh)
                self.evalmesh.append(integ.evalmesh)

        for mesh in self.evalmesh:
            uniques,indices = np.unique(mesh[0],return_index=True)
            if len(indices) < len(mesh[0]):
                # to preserve initial order
                mask = np.zeros(len(mesh[0]),dtype='?')
                mask[indices] = True
                mesh[0] = mesh[0][mask]

        for projname,integ in self.integration_engines.items():
            cmesh = self.evalmesh[integ.indexmesh][0]
            integ.maskmesh = None
            if not np.all(cmesh == integ.evalmesh[0]):
                integ.maskmesh = utils.match1d(integ.evalmesh[0],cmesh)[0]
                assert len(integ.maskmesh) == len(integ.evalmesh[0])

    def set_model_projection(self, data, projname, integration=None):
        integration = integration or {}
        #proj = ProjectionName(projname)
        x = data.get_x(proj=projname)
        error = ValueError('Unknown projection {} -> {}'.format(self.model_base.mode,projname.mode))
        self.shotnoise[projname] = 0.
        if self.model_base.space == ProjectionBase.POWER:
            if projname.mode == ProjectionName.MUWEDGE or (projname.mode == ProjectionName.MULTIPOLE and projname.proj == 0):
                self.shotnoise[projname] = self.model_base.get('shotnoise',0.)
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

    def __call__(self, fun, concatenate=True, remove_shotnoise=True, **kwargs):
        evals = [fun(*mesh,**kwargs) for mesh in self.evalmesh]
        toret = []
        for projname in self.projs:
            integ = self.integration_engines[projname]
            mesh = evals[integ.indexmesh]
            projected = integ(mesh[integ.maskmesh,...] if integ.maskmesh is not None else mesh) - self.shotnoise[projname] * remove_shotnoise
            toret.append(projected.flatten())
        if concatenate:
            return np.concatenate(toret)
        return toret

    def to_data_vector(self, fun, **kwargs):
        from cosmopipe.lib.data_vector import DataVector
        y = self(fun,concatenate=True,**kwargs)
        data = self.data.copy(copy_proj=True)
        data.set_y(y)
        return data
