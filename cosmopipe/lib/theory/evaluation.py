"""Evaluation of (callable) theory models at given points."""

import numpy as np

from cosmopipe.lib.utils import BaseClass
from cosmopipe.lib import utils
from cosmopipe.lib.data_vector import DataVector, ProjectionName, ProjectionNameCollection

from .base import ProjectionBasis, ProjectionBasisCollection
from .integration import MultipoleIntegration, MuWedgeIntegration, MultipoleToMultipole, MultipoleToMuWedge


class ModelEvaluation(BaseClass):

    """Class evaluating theory model(s) at given points, optionally performing some integration (e.g. against Legendre polynomials)."""

    def __init__(self, data_vector, projs=None, model_bases=None, integration=None):
        """
        Initialize :class:`ModelEvaluation`.

        Parameters
        ----------
        data_vector : DataVector
            Data vector to project onto.

        projs : list, ProjectionNameCollection, default=None
            Projection names.
            If ``None``, defaults to ``data_vector.get_projs()``, i.e. projections within data view.
            Theory model(s) are matched to each of these projections (using :meth:`ProjectionBasisCollection.get_by_proj`)

        model_bases : ProjectionBasis, ProjectionBasisCollection
            Projection basis of input model(s).
            If single projection basis, a single model is expected in :meth:`__call__`.

        integration : dict
            Options for model integration. See :meth:`set_model_evaluation`.
        """
        self.single_model = not isinstance(model_bases,(list,ProjectionBasisCollection))
        self.model_bases = ProjectionBasisCollection(model_bases)
        self.data_vector = data_vector
        if not isinstance(data_vector,DataVector):
            self.data_vector = DataVector(data_vector,proj=projs)
        self.projs = ProjectionNameCollection(projs) if projs is not None else data_vector.get_projs()

        if integration is None:
            integration = {proj:None for proj in self.projs}
        self.integration_options = integration

        self.integration_engines = {}
        groups = {}
        # Group projections using the same model basis
        for proj in self.projs:
            model_basis = self.model_bases.get_by_proj(proj)
            if model_basis not in groups:
                groups[model_basis] = []
            groups[model_basis].append(proj)
            self.set_model_evaluation(self.data_vector,proj,model_basis=model_basis,integration=self.integration_options[proj])

        self.evalmesh = {}
        for model_basis in groups:
            projs = groups[model_basis]
            self.evalmesh[model_basis] = [[mesh.copy() for mesh in self.integration_engines[projs[0]].evalmesh]]
            for proj in projs:
                integ = self.integration_engines[proj]
                integ.indexmesh = None
                for imesh,mesh in enumerate(self.evalmesh[model_basis]):
                    if all(np.all(m1 == m2) for m1,m2 in zip(mesh[1:],integ.evalmesh[1:])):
                        mesh[0] = np.concatenate([mesh[0],integ.evalmesh[0]])
                        integ.indexmesh = imesh
                if integ.indexmesh is None:
                    integ.indexmesh = len(self.evalmesh[model_basis])
                    self.evalmesh[model_basis].append(integ.evalmesh)

            for mesh in self.evalmesh[model_basis]:
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
                cmesh = self.evalmesh[model_basis][integ.indexmesh][0]
                integ.maskmesh = None
                if not np.all(cmesh == integ.evalmesh[0]):
                    integ.maskmesh = utils.match1d(integ.evalmesh[0],cmesh)[0]
                    assert len(integ.maskmesh) == len(integ.evalmesh[0])

    def set_model_evaluation(self, data_vector, proj, model_basis, integration=None):
        r"""
        Set engine for model evaluation on projection ``proj``.

        Parameters
        ----------
        data_vector : DataVector
            Data vector to project onto.

        proj : ProjectionName
            Projection name to project onto.

        model_basis : ProjectionBasis
            Model projection basis.

        integration : dict
            Options for model integration:

            - :class:`MultipoleIntegration` if input model lives in :math:`(x,\mu)` space
              and ``proj`` is a multipole.
            - :class:`MuWedgeIntegration` if input model lives in :math:`(x,\mu)` space
              and ``proj`` is a :math:`\mu`-wedge.
        """
        integration = integration or {}
        #proj = ProjectionName(proj)
        x = data_vector.get_x(proj=proj)
        error = ValueError('Unknown projection {} -> {}'.format(model_basis.mode,proj.mode))
        if proj.mode == ProjectionName.MULTIPOLE:
            if model_basis.mode == ProjectionBasis.MUWEDGE:
                self.integration_engines[proj] = MultipoleIntegration({**integration,'ells':(proj.proj,)})
                self.integration_engines[proj].evalmesh = [x,self.integration_engines[proj].mu]
            elif model_basis.mode == ProjectionBasis.MULTIPOLE:
                self.integration_engines[proj] = MultipoleToMultipole(ellsin=model_basis.projs,ellsout=(proj.proj,))
                self.integration_engines[proj].evalmesh = [x]
            else:
                raise error
        elif proj.mode == ProjectionName.MUWEDGE:
            if model_basis.mode == ProjectionBasis.MUWEDGE:
                self.integration_engines[proj] = MuWedgeIntegration({**integration,'muwedges':(proj.proj,)})
                self.integration_engines[proj].evalmesh = [x,self.integration_engines[proj].mu[0]]
            elif model_basis.mode == ProjectionBasis.MULTIPOLE:
                self.integration_engines[proj] = MultipoleToMuWedge(ellsin=model_basis.projs,muwedges=(proj.proj,))
                self.integration_engines[proj].evalmesh = [x]
            else:
                raise error
        else:
            raise error
        shotnoise = 0.
        if model_basis.space == ProjectionBasis.POWER:
            if proj.mode == ProjectionName.MUWEDGE or (proj.mode == ProjectionName.MULTIPOLE and proj.proj == 0):
                shotnoise = model_basis.get('shotnoise',0.)
        self.integration_engines[proj].model_basis = model_basis
        self.integration_engines[proj].shotnoise = shotnoise


    def __call__(self, models, concatenate=True, remove_shotnoise=True, **kwargs):
        """
        Project input models onto data vector, returning y-coordinates.

        Parameters
        ----------
        models : BaseModel, ModelCollection
            Model callable(s).
            A single model is expected if a single model basis was provided in :meth:`__init__`.

        concatenate : bool, default=True
            If ``True``, concatenates output y-coordinates over all projections.
            Else, returns a list of arrays corresponding to each :attr:`projs`.

        remove_shotnoise : bool, default=True
            For power spectrum models (``model.basis.space`` is ``'power'``), remove ``model.basis.shotnoise`` (defaults to 0).

        kwargs : dict
            Arguments for input model(s).

        Returns
        -------
        toret : list, array
            y-coordinates.
            If input ``concatenate`` is ``True``, single array concatenated over all projections.
            Else, a list of arrays corresponding to each :attr:`projs`.
        """
        evals = {}
        for model_basis in self.evalmesh:
            evals[model_basis] = []
            model = models if self.single_model else models.get(model_basis)
            for mesh in self.evalmesh[model_basis]:
                evals[model_basis].append(model(*mesh,**kwargs))
        toret = []
        for proj in self.projs:
            integ = self.integration_engines[proj]
            mesh = evals[integ.model_basis][integ.indexmesh]
            projected = integ(mesh[integ.maskmesh,...] if integ.maskmesh is not None else mesh) - integ.shotnoise * remove_shotnoise
            toret.append(projected.flatten())
        if concatenate:
            return np.concatenate(toret)
        return toret

    def to_data_vector(self, models, **kwargs):
        """Same as :meth:`__call__`, but returning :class:`DataVector` instance."""
        y = self(models,concatenate=True,**kwargs)
        data_vector = self.data_vector.deepcopy()
        data_vector.set_y(y,proj=self.projs)
        return data_vector

    @classmethod
    def propose_out(cls, model_bases=None):
        """
        Propose output projection names given proposed input projection names ``projsin``.

        Parameters
        ----------
        projsin : list, ProjectionNameCollection
            Input projections.

        model_bases : ProjectionBasis, ProjectionBasisCollection
            Projection basis of input model(s).

        Returns
        -------
        toret : ProjectionNameCollection
            Proposed projection names.
        """
        model_bases = ProjectionBasisCollection(model_bases)
        toret = ProjectionNameCollection()
        for model_basis in model_bases:
            if model_basis.mode == ProjectionBasis.MUWEDGE and model_basis.projs is None:
                model_basis = model_basis.copy()
                muwedges = np.linspace(0.,1.,4)
                model_basis.projs = list(zip(muwedges[:-1],muwedges[1:]))
            projs = model_basis.to_projs()
            toret += projs
            from cosmopipe.lib.survey_selection.projection import ProjectionConversion
            if model_basis.mode == ProjectionBasis.MULTIPOLE:
                baseout = model_basis.copy(mode=ProjectionBasis.MUWEDGE,projs=None)
                toret += ProjectionConversion.propose_out(projs,baseout=baseout)
            if model_basis.mode == ProjectionBasis.MUWEDGE:
                baseout = model_basis.copy(mode=ProjectionBasis.MULTIPOLE,projs=None)
                toret += ProjectionConversion.propose_out(projs,baseout=baseout)
        return toret
