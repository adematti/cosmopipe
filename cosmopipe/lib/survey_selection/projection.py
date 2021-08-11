"""Classes to link all (linear) survey selection effects together."""

import numpy as np

from cosmopipe.lib.utils import BaseClass
from cosmopipe.lib.data_vector import DataVector, ProjectionName, ProjectionNameCollection
from cosmopipe.lib.theory import ProjectionBasis, ProjectionBasisCollection, ModelEvaluation

from .base import ProjectionConversion, BaseRegularMatrix
from .binning import BaseBinning


class ModelProjection(BaseClass):
    """
    Class projecting theory model onto data vector by chaining several matrix operations,
    accounting for the various survey selection effects: wide-angle, window function etc.

    Attributes
    ----------
    model_bases : ProjectionBasisCollection
        Projection bases of input theory models.

    data : DataVector
        Data vector to project onto.

    projs : ProjectionNameCollection
        Projection names.

    operations : list
        List of successive matrix operations to apply.
    """
    def __init__(self, data, projs=None, model_bases=None, integration=None):
        """
        Initialize :class:`ModelProjection`.

        Parameters
        ----------
        data : DataVector
            Data vector to project onto.

        projs : list, ProjectionNameCollection, default=None
            Projection names.
            If ``None``, defaults to ``data.get_projs()``, i.e. projections within data view.

        model_bases : ProjectionBasis, ProjectionBasisCollection
            Projection basis of input model(s).
            If single projection basis, a single model is expected in :meth:`__call__`.

        integration : dict
            Options for model evaluation / integration, see :class:`ModelEvaluation`.
        """
        if isinstance(data,self.__class__):
            self.__dict__.update(data.__dict__)
            return

        self.single_model = not isinstance(model_bases,(list,ProjectionBasisCollection))
        self.model_bases = ProjectionBasisCollection(model_bases)

        self.data = data
        self.projs = ProjectionNameCollection(projs) if projs is not None else data.get_projs()
        self.operations = []
        self.integration_options = integration

    def copy(self, copy_operations=True):
        """
        Return shallow-copy of model projection, including internal list of operations :attr:`operations`.
        If ``copy_operations`` is ``True``, further shallow-copy each operation.
        """
        new = self.__copy__()
        if copy_operations:
            new.operations = [operation.copy() for operation in self.operations]
        else:
            new.operations = self.operations.copy()
        return new

    def insert(self, index, operation):
        """Insert operation at index ``index`` in list of operations :attr:`operations`."""
        self.operations.insert(index,operation)

    def append(self, operation):
        """Append operation to the list of operations :attr:`operations`."""
        self.operations.append(operation)

    def setup(self, data=None, projs=None):
        """
        Compile all operations of :attr:`operations` into a single matrix transform.

        Parameters
        ----------
        data : DataVector, default=None
            Data vector to project onto. If not ``None``, replace current :attr:`data`.

        projs : list, ProjectionNameCollection
            Projection names. If not ``None``, replace current :attr:`projs`.
        """
        if data is not None:
            self.data = data
        if projs is not None:
            self.projs = ProjectionNameCollection(projs)
        operations = self.operations.copy()
        if not operations or not isinstance(operations[-1],BaseBinning):
            binning = BaseBinning()
            operations.append(binning)

        # this is binning operation
        operations[-1].setup(data=self.data,projs=self.projs)

        operation = operations[0]
        projsout = ModelEvaluation.propose_out(self.model_bases)
        projsin,projsout = operation.propose_out(projsout)
        projsins = [projsin]
        projsouts = [projsout]
        for operation in operations[1:-1]:
            projsout = ProjectionConversion.propose_out(projsout,baseout=operation.basis)
            projsin,projsout = operation.propose_out(projsout) # here we can change projsin, projsout
            projsins.append(projsin)
            projsouts.append(projsout)

        self.matrix = operations[-1].matrix # binning
        for projsin,projsout,operation,next in zip(projsins[::-1],projsouts[::-1],operations[-2::-1],operations[:0:-1]):
            if operation.regularout:
                if not next.regularin:
                    raise ValueError('An interpolation must be performed between {} and {}'.format(operation.__class__.__name__,next.__class__.__name__))
                conversion = ProjectionConversion(next.xin,projsout=next.projsin,projsin=projsout)
                x = conversion.xin
                projsout = conversion.projsin
                self.matrix = self.matrix @ conversion.matrix
            else:
                x = next.xin
                projsout = next.projsin
            operation.setup(x,projsin=projsin,projsout=projsout) # setup can reduce projsin
            self.matrix = self.matrix @ operation.matrix
        if np.allclose(self.matrix,np.eye(*self.matrix.shape,dtype=self.matrix.dtype)):
            self.matrix = None

        operation = operations[0]
        data = DataVector(x=operation.xin,proj=operation.projsin)

        # evaluation of the model callable
        self.evaluation = ModelEvaluation(data=data,model_bases=self.model_bases[0] if self.single_model else self.model_bases,integration=self.integration_options)

        xout = operations[-1].xout
        cumsizes = np.cumsum([0] + [len(x) for x in xout])
        self.slices = [slice(start,stop) for start,stop in zip(cumsizes[:-1],cumsizes[1:])]

    def __call__(self, models, concatenate=True, **kwargs):
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

        kwargs : dict
            Arguments for :meth:`ModelEvaluation.__call__`.

        Returns
        -------
        toret : list, array
            y-coordinates.
            If input ``concatenate`` is ``True``, single array concatenated over all projections.
            Else, a list of arrays corresponding to each :attr:`projs`.
        """
        toret = self.evaluation(models,concatenate=True,remove_shotnoise=True,**kwargs)
        if self.matrix is not None:
            toret = self.matrix.dot(toret)
        if not concatenate:
            return [toret[sl] for sl in self.slices]
        return toret

    def to_data_vector(self, models, **kwargs):
        """Same as :meth:`__call__`, but returning :class:`DataVector` instance."""
        y = self(models,concatenate=True,**kwargs)
        data = self.data.deepcopy()
        #print(y.size)
        data.set_y(y,proj=self.projs)
        return data


class ModelProjectionCollection(BaseClass):
    """
    Class managing several model projections.
    Required data projections are grouped by e.g. :attr:`ProjectionName.space`, :attr:`ProjectionName.name`
    (every :class:`ProjectionName` attribute that is not :attr:`ProjectionName.mode`, :attr:`ProjectionName.proj`, :attr:`ProjectionName.wa_order`)
    and :class:`ModelProjection` instances are created for each of these groups.

    Attributes
    ----------
    data : DataVector
        Data vector to project onto.

    projs : ProjectionNameCollection
        Projection names.

    model_projections : list
        List of :class:`ModelProjection` instances.
    """
    def __init__(self, data, projs=None, model_bases=None, integration=None):
        """
        Initialize :class:`ModelProjectionCollection`.

        Parameters
        ----------
        data : DataVector
            Data vector to project onto.

        projs : list, ProjectionNameCollection, default=None
            Projection names.
            If ``None``, defaults to ``data.get_projs()``, i.e. projections within data view.

        model_bases : ProjectionBasis, ProjectionBasisCollection
            Projection basis of input model(s).
            If single projection basis, a single model is expected in :meth:`__call__`.

        integration : dict
            Options for model evaluation / integration, see :class:`ModelEvaluation`.
        """
        self.data = data
        self.projs = ProjectionNameCollection(projs) if projs is not None else data.get_projs()

        self.groups = self.projs.group_by(exclude=['mode','proj','wa_order'])
        self.projection_mapping = [None]*len(self.projs)
        nprojs = 0
        self.model_projections = []
        for label,projs in self.groups.items():
            indices = [self.projs.index(proj) for proj in projs]
            for ii,jj in enumerate(indices): self.projection_mapping[jj] = nprojs + ii
            nprojs += len(projs)
            modelproj = ModelProjection(self.data,projs=projs,model_bases=model_bases,integration=integration)
            self.model_projections.append(modelproj)

    def copy(self, *args, **kwargs):
        """Return copy of collection of model projections, including copy of each model projections, with arguments ``args`` and ``kwargs``."""
        new = self.__copy__()
        new.model_projections = [modelproj.copy(*args,**kwargs) for modelproj in self.model_projections]
        return new

    def set(self, *args, **kwargs):
        """
        Set new :class:`ModelProjection` instance, initialized with arguments ``args`` and ``kwargs``.
        Projections which are already covered by current collection are *not* replaced.
        TODO: this was to simplify things, but we should simply redo the :attr:`projection_mapping`.
        """
        modelproj = ModelProjection(*args,**kwargs)
        add_modelproj = False
        for proj in modelproj.projs:
            add_modelproj = proj not in self.projs
            if add_modelproj:
                self.projs.set(proj)
                self.projection_mapping.append(len(self.projs))
        if add_modelproj:
            self.data += modelproj.data
            self.model_projections.append(modelproj)

    def setup(self, data=None, projs=None):
        """
        Set up all model projections.

        Parameters
        ----------
        data : DataVector, default=None
            Data vector to project onto. If not ``None``, replace current :attr:`data`.

        projs : list, ProjectionNameCollection
            Projection names. If not ``None``, replace current :attr:`projs`.
        """
        if data is not None:
            self.data = data

        if data is not None or projs is not None:
            self_group_labels = list(self.groups.keys())
            self_model_projections = self.model_projections
            self.projs = ProjectionNameCollection(projs) if projs is not None else data.get_projs()
            self.groups = self.projs.group_by(exclude=['mode','proj','wa_order'])
            self.projection_mapping = [None]*len(self.projs)
            nprojs = 0
            self.model_projections = []
            for label,projs in self.groups.items():
                indices = [projs.index(proj) for proj in projs]
                for ii,jj in enumerate(indices): self.projection_mapping[jj] = nprojs + ii
                self.model_projections.append(self_model_projections[self_group_labels.index(label)])

        for model_projection,projs in zip(self.model_projections,self.groups.values()):
            model_projection.setup(data=data,projs=projs)

    @classmethod
    def concatenate(cls, *others):
        """Concatenate model projection collections."""
        new = others[0].copy()
        for other in others[1:]:
            for item in other.model_projections:
                new.set(item)
        return new

    def extend(self, other):
        """Extend collection with ``other``."""
        new = self.concatenate(self,other)
        self.__dict__.update(new.__dict__)

    def __radd__(self, other):
        """Operation corresponding to ``other + self``."""
        if other in [[],0,None]:
            return self.copy()
        return self.concatenate(self,other)

    def __iadd__(self, other):
        """Operation corresponding to ``self += other``."""
        self.extend(other)
        return self

    def __add__(self, other):
        """Addition of two collections is defined as concatenation."""
        return self.concatenate(self,other)

    def __call__(self, models, concatenate=True, **kwargs):
        """
        Project input model(s) onto data vector, returning y-coordinates.

        Parameters
        ----------
        models : BaseModel, ModelCollection
            Model callable(s).
            A single model is expected if a single model basis was provided in :meth:`__init__`.

        concatenate : bool, default=True
            If ``True``, concatenates output y-coordinates over all projections.
            Else, returns a list of arrays corresponding to each :attr:`projs`.

        kwargs : dict
            Arguments for :meth:`ModelEvaluation.__call__`

        Returns
        -------
        toret : list, array
            y-coordinates.
            If input ``concatenate`` is ``True``, single array concatenated over all projections.
            Else, a list of arrays corresponding to each :attr:`projs`.
        """
        tmp = []
        for model_projection in self.model_projections:
            tmp += model_projection(models,concatenate=False, **kwargs)
        toret = [tmp[ii] for ii in self.projection_mapping]
        if concatenate:
            return np.concatenate(toret)
        return toret

    def to_data_vector(self, models, **kwargs):
        """Same as :meth:`__call__`, but returning :class:`DataVector` instance."""
        y = self(models,concatenate=True,**kwargs)
        data = self.data.deepcopy()
        data.set_y(y,proj=self.projs)
        return data
