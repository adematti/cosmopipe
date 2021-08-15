"""Base classes to handle theory models: model basis and collections."""

import functools

import numpy as np
from scipy import interpolate
from cosmoprimo import PowerSpectrumInterpolator1D

from cosmopipe.lib.utils import BaseClass, BaseNameSpace, BaseOrderedCollection
from cosmopipe.lib.data_vector import ProjectionName, ProjectionNameCollection
from .fog import get_FoG


class ProjectionBasis(BaseNameSpace):
    r"""
    Class representing a model basis.

    Attributes
    ----------
    x : array
        Array of x-coordinates where the model can be safely evaluated.

    name : string
        Model name.

    fields : tuple
        Tracer field(s).

    space : string
        Projection space, e.g. power spectrum ('power')? Correlation function ('correlation')?

    shotnoise : float
        If in Fourier space, Poisson shotnoise associated to that model.

    mode : string
        Projection mode, e.g. 'multipole'? 'muwedge'?

    projs : list
        List of projection number or identifier (e.g. order of Legendre polynomial, lower and upper limit of :math:`\mu`-wedge).

    wa_order : int
        Wide-angle order.
    """
    MULTIPOLE = 'multipole'
    MUWEDGE = 'muwedge'
    POWER = 'power'
    CORRELATION = 'correlation'
    _attrs = ['x','name','fields','space','shotnoise','mode','projs','wa_order']

    def __init__(self, x=None, **kwargs):
        """
        Initialize :class:`ProjectionBasis`.

        Parameters
        ----------
        x : array, dict, default=None
            Dictionary of attribute values,
            or array of x-coordinates where the model can be safely evaluated.

        kwargs : dict
            Dictionary of attribute values.
            If ``x`` is dictionary, is updated by ``kwargs``.
        """
        if isinstance(x,self.__class__):
            self.__dict__.update(x.__dict__)
            return
        for name in self._attrs:
            setattr(self,name,None)
        if isinstance(x,dict):
            self.__init__(**x)
            self.set(**kwargs)
            return
        self.x = x
        self.set(**kwargs)

    def __repr__(self):
        """String representation including only attributes with non-``None`` values, and droping :attr:`x` array."""
        toret = ['{}={}'.format(name,value) for name,value in self.as_dict(drop_none=True).items() if name != 'x']
        return '{}({})'.format(self.__class__.__name__,','.join(toret))

    def __gt__(self, other):
        # Used for sorting
        return np.mean(self.projs) > np.mean(other.projs)

    def __lt__(self, other):
        # Used for sorting
        return np.mean(self.projs) < np.mean(other.projs)

    def __hash__(self):
        # If __eq__ is redefined, __hash__ must be as well
        return hash(self.name)

    def __eq__(self, other):
        """
        Is ``self`` equal to ``other``, i.e. same type and attributes?
        Does not perform comparison of :attr:`x` array.
        """
        return isinstance(other,self.__class__) and all(getattr(self,name) == getattr(other,name) for name in self._attrs if name != 'x')

    def to_projs(self):
        """Export projection basis as a :class:`ProjectionNameCollection` instance."""
        toret = ProjectionNameCollection()
        di = {name:value for name,value in self.as_dict(drop_none=True).items() if name in ProjectionName._attrs and name != 'proj'}
        return ProjectionNameCollection([ProjectionName(proj=proj,**di) for proj in self.projs])


class ProjectionBasisCollection(BaseOrderedCollection):
    """
    Class representing a collection of projection basis.

    Note
    ----
    When adding a basis equal to another already in the collection, the latter will be replaced by the former.
    Insertion order is conserved.
    """
    _cast = lambda x: x if isinstance(x,ProjectionBasis) else ProjectionBasis(x)

    def spaces(self):
        """Return list of :attr:`ProjectionBasis.space`."""
        return [basis.space for data in self.data]

    def modes(self):
        """Return list of :attr:`ProjectionBasis.mode`."""
        return [basis.mode for mode in self.data]

    def get_by_proj(self, *args, **kwargs):
        """
        Try to fetch the projection basis that matches the input :class:`ProjectionName`, by priority order:

        - same :attr:`ProjectionBasis.name`
        - same :attr:`ProjectionBasis.space`
        - same :attr:`ProjectionBasis.mode`
        - same :attr:`ProjectionBasis.wa_order`

        Stops as soon as a single match is found.
        Raises an :class:`IndexError` if no or several matches are found.

        Parameters
        ----------
        args : list
            Arguments for :class:`ProjectionName`.

        kwargs : dict
            Arguments for :class:`ProjectionName`.

        Returns
        -------
        basis : ProjectionBasis
        """
        proj = ProjectionName(*args,**kwargs)
        indices = range(len(self))
        if proj.name is not None:
            indices = [index for index in indices if self.data[index].name == proj.name]
        if proj.space is not None:
            indices = [index for index in indices if self.data[index].space == proj.space]
        if len(indices) > 1 and proj.mode is not None:
            indices = [index for index in indices if self.data[index].mode == proj.mode]
        if len(indices) > 1 and proj.wa_order is not None:
            indices = [index for index in indices if self.data[index].wa_order == proj.wa_order]
        if not len(indices):
            raise IndexError('Could not find any match between data projection {} and model bases {}'.format(proj,self))
        if len(indices) > 1:
            raise IndexError('Data projection {} corresponds to several model bases {}'.format(proj,[self.data[ii] for ii in indices]))
        return self.data[indices[0]]


class ModelCollection(BaseOrderedCollection):
    """
    Class representing a collection of models.

    Note
    ----
    When adding a model with same :class:`ProjectionBasis` than another already in the collection, the latter will be replaced by the former.
    Insertion order is conserved.
    """
    _cast = lambda x: x if isinstance(x,ProjectionBasis) else x.basis if isinstance(x,BaseModel) else ProjectionBasis(x) # for __contains__

    def bases(self):
        """Return model bases as :class:`ProjectionBasisCollection`."""
        return ProjectionBasisCollection([model.basis for model in self.data])

    @property
    def models(self):
        """Return models."""
        return self.data

    def __repr__(self):
        """String representation."""
        return '{}({})'.format(self.__class__.__name__,repr(self.data))

    def index(self, basis):
        """Return index of model with :attr:`ProjectionBasis` ``basis``."""
        basis = self.__class__._cast(basis)
        return self.bases().index(basis)

    def get(self, basis):
        """Return model with :attr:`ProjectionBasis` ``basis``."""
        return self.data[self.index(basis)]

    def set(self, model):
        """Set input model (with ``basis`` attribute)."""
        if not hasattr(model,'basis'):
            raise ValueError('Input model must have a `basis` attribute.')
        if model in self:
            self.data[self.index(model)] = model
        else:
            self.data.append(model)

    def __contains__(self, item):
        """Whether collection contains this model or basis."""
        return self.__class__._cast(item) in self.bases()

    def __getitem__(self, index):
        """Return model by index."""
        return self.data[index]

    def select(self, *args, **kwargs):
        """
        Return new collection, after basis selection.
        Same arguments (:class:`ProjectionBasis` attributes) as :meth:`ProjectionBasisCollection.select`.
        """
        new = self.__class__()
        bases = self.bases().select(*args,**kwargs)
        for basis in bases:
            new.set(self.get(basis))
        return new

    def items(self):
        """Yield tuples of model basis and model."""
        for model in self.data:
            yield (model.basis, model)

    def get_by_proj(self, *args, **kwargs):
        """
        Return model corresponding to :class:`ProjectionName`.
        Same arguments as :meth:`ProjectionBasisCollection.get_by_proj`.
        """
        basis = self.bases().get_by_proj(*args,**kwargs)
        return self.get(basis)


class BaseModel(BaseClass):
    """
    Base class to represent theory model.

    Attributes
    ----------
    basis : ProjectionBasis
        Basis for this model.
    """
    def __init__(self, basis=None):
        """
        Initialize :class:`BaseModel`.

        Parameters
        ----------
        basis : dict, ProjectionBasis
            Projection basis.
        """
        self.basis = ProjectionBasis(basis or {})

    def __call__(self, *args, **kwargs):
        """Evaluate model."""
        return self.eval(*args,**kwargs)


class BasePTModel(BaseModel):

    """Base class to represent perturbation theory model."""

    def __init__(self, pklin, klin=None, FoG='gaussian'):
        """
        Initialize :class:`BasePTModel`.

        Parameters
        ----------
        pklin : array, callable, PowerSpectrumInterpolator1D
            Linear power spectrum. If array, ``klin`` must be provided.

        klin : array, default=None
            Wavenumbers. Must be provided if ``pklin`` is an array.

        FoG : callable, string, default='gaussian'
            Type of Finger-of-God effect.
        """
        if callable(pklin):
            self.pk_linear = pklin
            if klin is None: klin = getattr(pklin,'k',None)
        elif klin is not None:
            self.pk_linear = PowerSpectrumInterpolator1D(k=klin,pk=pklin)
        else:
            raise ValueError('Input pklin should be a PowerSpectrumInterpolator1D instance if no k provided.')
        self.FoG = FoG
        if FoG is not None:
            self.FoG = get_FoG(FoG)
        self.basis = ProjectionBasis(x=klin,space=ProjectionBasis.POWER,mode=ProjectionBasis.MUWEDGE,wa_order=0)
