"""Classes to link all (linear) survey selection effects together."""

import numpy as np

from cosmopipe.lib.utils import BaseClass, BaseOrderedCollection
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

    data_vector : DataVector
        Data vector to project onto.

    projs : ProjectionNameCollection
        Projection names.

    operations : list
        List of successive matrix operations to apply.
    """
    def __init__(self, data_vector, projs=None, model_bases=None, integration=None):
        """
        Initialize :class:`ModelProjection`.

        Parameters
        ----------
        data_vector : DataVector
            Data vector to project onto.

        projs : list, ProjectionNameCollection, default=None
            Projection names.
            If ``None``, defaults to ``data_vector.get_projs()``, i.e. projections within data view.

        model_bases : ProjectionBasis, ProjectionBasisCollection
            Projection basis of input model(s).
            If single projection basis, a single model is expected in :meth:`__call__`.

        integration : dict
            Options for model evaluation / integration, see :class:`ModelEvaluation`.
        """
        if isinstance(data_vector,self.__class__):
            self.__dict__.update(data_vector.__dict__)
            return

        self.single_model = not isinstance(model_bases,(list,ProjectionBasisCollection))
        self.model_bases = ProjectionBasisCollection(model_bases)

        self.set_data_vector(data_vector)
        self.set_projs(projs)

        self.operations = []
        self.integration_options = integration

    def set_data_vector(self, data_vector):
        """Set :attr:`data_vector`."""
        self.data_vector = data_vector

    def set_projs(self, projs):
        """Set :attr:`projs`."""
        # we want projections to be exactly the same as data for ModelProjectionCollection
        if projs is not None:
            self.projs = ProjectionNameCollection([self.data_vector.projs.get(proj) for proj in projs])
        else:
            self.projs = self.data_vector.get_projs()

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

    def setup(self, data_vector=None, projs=None):
        """
        Compile all operations of :attr:`operations` into a single matrix transform.

        Parameters
        ----------
        data_vector : DataVector, default=None
            Data vector to project onto. If not ``None``, replace current :attr:`data_vector`.

        projs : list, ProjectionNameCollection
            Projection names. If not ``None``, replace current :attr:`projs`.
        """
        if data_vector is not None:
            self.set_data_vector(data_vector)
        if projs is not None:
            self.set_projs(projs)
        operations = self.operations.copy()
        if not operations or not isinstance(operations[-1],BaseBinning):
            binning = BaseBinning()
            operations.append(binning)

        # this is binning operation
        operations[-1].setup(data_vector=self.data_vector,projs=self.projs)

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
        data_vector = DataVector(x=operation.xin,proj=operation.projsin)

        # evaluation of the model callable
        self.evaluation = ModelEvaluation(data_vector=data_vector,model_bases=self.model_bases[0] if self.single_model else self.model_bases,integration=self.integration_options)

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
        data_vector = self.data_vector.deepcopy()
        #print(y.size)
        data_vector.set_y(y,proj=self.projs)
        return data_vector


class ModelProjectionCollection(BaseOrderedCollection):
    """
    Class managing a collection of model projections.
    This is useful to directly get the the theory prediction of heteregenous data,
    hence using different :class:`ModelProjection` pipelines,
    e.g. correlation function and power spectrum.

    Attributes
    ----------
    _mapping : list, None
        If not ``None``, this list contains, in the desired order,
        the index of each projection in the concatenated projection collection from all model projections.
        This is useful if one wants :meth:`__call__` or :attr:`to_data_vector` to return
        e.g. a projection of the second model projection *before* one of the first model projection.
        This mapping can be set using :meth:`reorder_by_projs`.
    """
    def __init__(self, *args, **kwargs):
        """Initialize :class:`ModelProjectionCollection`."""
        super(ModelProjectionCollection,self).__init__(*args,**kwargs)
        self._mapping = None

    def index(self, proj):
        """Return index of :class:`ProjectionName` ``proj`` in collection."""
        proj = ProjectionName(proj)
        for ii,model_projection in enumerate(self.data):
            indices = model_projection.projs.index(proj,ignore_none=True)
            if len(indices) == 1: break
            elif len(indices) > 1: raise IndexError('Data projection {} has several matches')
        return ii

    def set(self, model_projection):
        """
        Set new :class:`ModelProjection` in collection.
        Raise :class:`TypeError` if not of type :class:`ModelProjection`.
        """
        #Raise :class:`IndexError` if this model projection covers projections that are already treated in a model projection of this collection.
        if not isinstance(model_projection,ModelProjection):
            raise TypeError('{} is not a ModelProjection instance.'.format(model_projection))
        # removing already-covered projections
        for self_model_projection in self.data:
            new_projs = self_model_projection.projs.copy()
            for proj in model_projection.projs:
                if proj in new_projs: del new_projs[new_projs.index(proj)]
            self_model_projection.set_projs(new_projs)
        #for proj in model_projection.projs:
        #    if proj in self.projs:
        #        raise ValueError('Data projection {} is already in current {}'.format(proj,self.__class__.__name__))
        self.data.append(model_projection)

    @property
    def data_vector(self):
        """Return data vector, as a concatenation of all :attr:`ModelProjection.data_vector`."""
        return DataVector.concatenate(*[model_projection.data_vector for model_projection in self])

    @property
    def projs(self):
        """Return projection names, as a concatenation of all :attr:`ModelProjection.projs`, optionally reordered."""
        projs = ProjectionNameCollection.concatenate(*[model_projection.projs for model_projection in self])
        if self._mapping is not None:
            projs.reorder(self._mapping)
        return projs

    @classmethod
    def propose_groups(cls, projs):
        """Group input projection names ``projs`` into sets that could be handled by a single :class:`ModelProjection`."""
        return ProjectionNameCollection(projs).group_by(exclude=['mode','proj','wa_order'])

    def reorder_by_projs(self, projs):
        """Internally reoder desired projections following projection names ``projs``."""
        allprojs = ProjectionNameCollection.concatenate(*[model_projection.projs for model_projection in self])
        self._mapping = [allprojs.index(proj) for proj in projs]

    def setup(self, data_vector=None, projs=None):
        """
        Set up all model projections.

        Parameters
        ----------
        data_vector : DataVector, default=None
            Data vector to project onto. If not ``None``, replace current :attr:`data_vector`.

        projs : list, ProjectionNameCollection
            Projection names. If not ``None``, replace current :attr:`projs`.
        """
        if data_vector is not None or projs is not None:
            if projs is None: projs = data_vector.get_projs()
            else: projs = ProjectionNameCollection([data_vector.projs.get(proj) for proj in projs])
            indices = [self.index(proj) for proj in projs]
            for ii,model_projection in enumerate(self.data):
                if ii not in indices: continue
                model_projection_projs = [proj for proj,ind in zip(projs,indices) if ind == ii]
                model_projection.setup(data_vector=data_vector,projs=model_projection_projs)
            self.reorder_by_projs(projs)

        else:
            for model_projection in self.data:
                model_projection.setup()

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
        toret = []
        for model_projection in self.data:
            toret += model_projection(models,concatenate=False,**kwargs)
        if self._mapping is not None:
            toret = [toret[ii] for ii in self._mapping]
        if concatenate:
            return np.concatenate(toret)
        return toret

    def to_data_vector(self, models, **kwargs):
        """Same as :meth:`__call__`, but returning :class:`DataVector` instance."""
        y = self(models,concatenate=True,**kwargs)
        data_vector = self.data_vector
        data_vector = data_vector.deepcopy()
        data_vector.set_y(y,proj=self.projs)
        return data_vector
