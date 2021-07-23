import numpy as np

from cosmopipe.lib.utils import BaseClass
from cosmopipe.lib.data_vector import DataVector, ProjectionName, ProjectionNameCollection

from .base import ProjectionBase, ProjectionBaseCollection
from .integration import MultipoleIntegration, MuWedgeIntegration, MultipoleToMultipole, MultipoleToMuWedge
from . import utils


class ModelEvaluation(BaseClass):

    def __init__(self, data, projs=None, model_base=None, integration=None):

        self.single_model = not isinstance(model_base,(list,ProjectionBaseCollection))
        self.model_bases = ProjectionBaseCollection(model_base)
        self.data = data
        if not isinstance(data,DataVector):
            self.data = DataVector(data,proj=projs)
        self.projs = ProjectionNameCollection(projs) if projs is not None else data.get_projs()

        if integration is None:
            integration = {proj:None for proj in self.projs}
        self.integration_options = integration

        self.integration_engines = {}
        groups = {}
        for proj in self.projs:
            model_base = self.model_bases.get_by_proj(proj)
            if model_base not in groups:
                groups[model_base] = []
            groups[model_base].append(proj)
            self.set_model_evaluation(self.data,proj,model_base=model_base,integration=self.integration_options[proj])

        self.evalmesh = {}
        for model_base in groups:
            projs = groups[model_base]
            self.evalmesh[model_base] = [[mesh.copy() for mesh in self.integration_engines[projs[0]].evalmesh]]
            for proj in projs:
                integ = self.integration_engines[proj]
                integ.indexmesh = None
                for imesh,mesh in enumerate(self.evalmesh[model_base]):
                    if all(np.all(m1 == m2) for m1,m2 in zip(mesh[1:],integ.evalmesh[1:])):
                        mesh[0] = np.concatenate([mesh[0],integ.evalmesh[0]])
                        integ.indexmesh = imesh
                if integ.indexmesh is None:
                    integ.indexmesh = len(self.evalmesh[model_base])
                    self.evalmesh[model_base].append(integ.evalmesh)

            for mesh in self.evalmesh[model_base]:
                uniques,indices = np.unique(mesh[0],return_index=True)
                if len(indices) < len(mesh[0]):
                    # to preserve initial order
                    #mask = np.zeros(len(mesh[0]),dtype='?')
                    #mask[indices] = True
                    #mesh[0] = mesh[0][mask]
                    mesh[0] = mesh[0][np.sort(indices)]

            # find match in concatenated mesh
            for proj in projs:
                integ = self.integration_engines[proj]
                cmesh = self.evalmesh[model_base][integ.indexmesh][0]
                integ.maskmesh = None
                if not np.all(cmesh == integ.evalmesh[0]):
                    integ.maskmesh = utils.match1d(integ.evalmesh[0],cmesh)[0]
                    assert len(integ.maskmesh) == len(integ.evalmesh[0])

    def set_model_evaluation(self, data, proj, model_base, integration=None):
        integration = integration or {}
        #proj = ProjectionName(projname)
        x = data.get_x(proj=proj)
        error = ValueError('Unknown projection {} -> {}'.format(model_base.mode,proj.mode))
        if proj.mode == ProjectionName.MULTIPOLE:
            if model_base.mode == ProjectionBase.MUWEDGE:
                self.integration_engines[proj] = MultipoleIntegration({**integration,'ells':(proj.proj,)})
                self.integration_engines[proj].evalmesh = [x,self.integration_engines[proj].mu]
            elif model_base.mode == ProjectionBase.MULTIPOLE:
                self.integration_engines[proj] = MultipoleToMultipole(ellsin=model_base.projs,ells=(proj.proj,))
                self.integration_engines[proj].evalmesh = [x]
            else:
                raise error
        elif proj.mode == ProjectionName.MUWEDGE:
            if model_base.mode == ProjectionBase.MUWEDGE:
                self.integration_engines[proj] = MuWedgeIntegration({**integration,'muwedges':(proj.proj,)})
                self.integration_engines[proj].evalmesh = [x,self.integration_engines[proj].mu[0]]
            elif model_base.mode == ProjectionBase.MULTIPOLE:
                self.integration_engines[proj] = MultipoleToMuWedge(ellsin=model_base.projs,muwedges=(proj.proj,))
                self.integration_engines[proj].evalmesh = [x]
            else:
                raise error
        else:
            raise error
        shotnoise = 0.
        if model_base.space == ProjectionBase.POWER:
            if proj.mode == ProjectionName.MUWEDGE or (proj.mode == ProjectionName.MULTIPOLE and proj.proj == 0):
                shotnoise = model_base.get('shotnoise',0.)
        self.integration_engines[proj].model_base = model_base
        self.integration_engines[proj].shotnoise = shotnoise


    def __call__(self, models, concatenate=True, remove_shotnoise=True, **kwargs):
        evals = {}
        for model_base in self.evalmesh:
            evals[model_base] = []
            model = models if self.single_model else models.get(model_base)
            for mesh in self.evalmesh[model_base]:
                evals[model_base].append(model(*mesh,**kwargs))
        toret = []
        for projname in self.projs:
            integ = self.integration_engines[projname]
            mesh = evals[integ.model_base][integ.indexmesh]
            projected = integ(mesh[integ.maskmesh,...] if integ.maskmesh is not None else mesh) - integ.shotnoise * remove_shotnoise
            toret.append(projected.flatten())
        if concatenate:
            return np.concatenate(toret)
        return toret

    def to_data_vector(self, models, **kwargs):
        y = self(models,concatenate=True,**kwargs)
        data = self.data.deepcopy()
        data.set_y(y,proj=self.projs)
        return data

    @classmethod
    def propose_out(cls, model_base=None):
        model_bases = ProjectionBaseCollection(model_base)
        toret = ProjectionNameCollection()
        for model_base in model_bases:
            if model_base.mode == ProjectionBase.MUWEDGE and model_base.projs is None:
                model_base = model_base.copy()
                muwedges = np.linspace(0.,1.,4)
                model_base.projs = list(zip(muwedges[:-1],muwedges[1:]))
            projs = model_base.to_projs()
            toret += projs
            from cosmopipe.lib.survey_selection.projection import ProjectionConversion
            if model_base.mode == ProjectionBase.MULTIPOLE:
                baseout = model_base.copy(mode=ProjectionBase.MUWEDGE,projs=None)
                toret += ProjectionConversion.propose_out(projs,baseout=baseout)
            if model_base.mode == ProjectionBase.MUWEDGE:
                baseout = model_base.copy(mode=ProjectionBase.MULTIPOLE,projs=None)
                toret += ProjectionConversion.propose_out(projs,baseout=baseout)
        return toret
