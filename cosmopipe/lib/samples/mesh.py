"""Definition of :class:`Mesh` to interpolate samples on a grid."""

import logging
import functools

import numpy as np

from cosmopipe.lib.utils import BaseClass, savefile
from cosmopipe.lib import mpi
from .utils import *


def sample_cic(edges, samples, weights=None):
    """
    Interpolate ``samples`` on mesh with Cloud-in-Cell (CIC) scheme.

    Parameters
    ----------
    edges: list
        List of mesh edges (one array for each mesh dimension).

    samples : array
        Array of positions, with coordinates along first axis.

    weights : array, default=None
        Sample weights, defaults to 1.

    Returns
    -------
    pdf : array
        Array with shape given by the length of edges along each dimension.
    """
    if not isinstance(edges,list): edges = [edges]
    if not isinstance(samples,list): samples = [samples]
    if weights is None: weights = np.ones_like(samples[0])
    pdf = np.zeros(tuple(map(len,edges)),dtype=edges[0].dtype)
    index,dindex = [],[]
    for g,d in zip(edges,samples):
        #i = np.clip(np.searchsorted(g,d,side='left')-1,0,len(g)-2)
        i = np.searchsorted(g,d,side='right')-1
        ok = (i>=0.) & ((i<=len(g)-2) | (d==g[-1]))
        weights[~ok] = 0.
        i[~ok] = 0
        i[i==len(g)-1] = len(g)-2
        index.append(i)
        di = (d-g[i])/(g[i+1]-g[i])
        #assert ((di>=0.) & (di<=1.)).all()
        dindex.append(di)
    index = np.array(index).T
    dindex = np.array(dindex).T
    ishifts = np.array(np.meshgrid(*([[0,1]]*len(samples)),indexing='ij')).reshape((len(samples),-1)).T
    for ishift in ishifts:
        sindex = index + ishift
        sweight = weights*np.prod((1-dindex) + ishift*(-1+2*dindex),axis=-1)
        #print sweight
        np.add.at(pdf,tuple(sindex.T),sweight)
    return pdf


def getaxes(func):
    """
    Wrapper that provides (list of) axes to wrapped method, from either (list of) axes or dimensions.
    """
    @functools.wraps(func)
    def wrapper(self, dims=None, axes=None, **kwargs):
        if axes is None:
            if dims is None:
                axes = tuple(range(self.ndim))
            else:
                axes = self.get_axes(dims)
        elif dims is not None:
            raise ValueError('Cannot provide both axes and dims.')
        return func(self,axes=axes,**kwargs)

    return wrapper


def getaxes_squeeze(func):
    """
    Wrapper that provides list of axes to wrapped method, from either (list of) axes or dimensions.
    If single axes or dimension provided as input, returns single output.
    """
    @functools.wraps(func)
    def wrapper(self, dims=None, axes=None, **kwargs):
        if axes is None:
            if dims is None:
                axes = tuple(range(self.ndim))
            else:
                axes = self.get_axes(dims)
        elif dims is not None:
            raise ValueError('Cannot provide both axes and dims.')
        isscalar = np.ndim(axes) == 0
        if isscalar:
            axes = [axes]
        toret = func(self,axes=axes,**kwargs)
        if isscalar:
            return toret[0]
        return toret

    return wrapper


class Mesh(BaseClass):
    """
    Class that represents a mesh.
    TODO: decide what to do with MPI... scatter mesh? Gather mesh?
    """

    @mpi.MPIInit
    def __init__(self, mesh, dims, edges=None, nodes=None, isdensity=True):
        """
        Initialize :class:`Mesh`.

        Parameters
        ----------
        mesh : array
            Mesh array.

        dims : list
            List of labels (one for each mesh dimension).

        edges : list, default=None
            List of edges (one array for each mesh dimension).
            If ``None``, use ``nodes`` instead.
            Mesh shape must be length of edges - 1 along each dimension.

        nodes : list, default=None
            List of mesh nodes (one array for each mesh dimension).

        isdensity : bool, default=True
            Whether to normalize binned (weighted) samples by mesh volume (``True``).
        """
        self.mesh = mesh
        self.dims = dims
        self.edges = edges
        self.nodes = nodes
        if nodes is not None and edges is None:
            centers = [(nodes[:-1] + nodes[1:])/2. for nodes in self.nodes]
            self.edges = [2*nodes[0] - centers[0]] + centers + [2*nodes[-1] - centers[-1]]
        elif nodes is None and edges is not None:
            self.nodes = [(edges[:-1] + edges[1:])/2. for edges in self.edges]
        if not isdensity:
            self.mesh /= self.volume()

    @property
    def shape(self):
        """Return mesh shape."""
        return tuple(len(edges)-1 for edges in self.edges)

    @property
    def ndim(self):
        """Return number of dimensions."""
        return len(self.edges)

    @getaxes_squeeze
    def widths(self, axes=None):
        """Return cell sizes along axes ``axes`` or dimensions ``dims``."""
        return [np.diff(self.edges[ax]) for ax in axes]

    @getaxes_squeeze
    def ranges(self, axes=None):
        """Return mesh extension along axes ``axes`` or dimensions ``dims``."""
        return [(self.edges[ax].min(),self.edges[ax].max()) for ax in axes]

    @getaxes_squeeze
    def mnodes(self, axes=None):
        """Return meshgrid of nodes along axes ``axes`` or dimensions ``dims``."""
        if axes is None: axes = range(self.ndim)
        isscalar = np.ndim(axes) == 0
        if isscalar:
            axes = [axes]
        toret = np.meshgrid(*[self.nodes[ax] for ax in axes],indexing='ij')
        if isscalar:
            return toret[0]
        return toret

    @getaxes_squeeze
    def volume(self, axes=None):
        """Return volume array along axes ``axes`` or dimensions ``dims``."""
        return np.prod(np.meshgrid(*self.widths(axes=axes),indexing='ij'),axis=0) # very memory inefficient...

    def get_axes(self, dims=None):
        """Return axes (axes) corresponding to dimensions ``dims``."""
        if dims is None: dims = self.dims
        isscalar = not isinstance(dims,(tuple,list))
        if isscalar:
            return self.dims.index(dims)
        return tuple(self.dims.index(dim) for dim in dims)

    @classmethod
    def from_samples(cls, samples, weights=None, dims=None, bins=30, method='cic', bw_method='scott'):
        """
        Initialize mesh from samples.
        TODO: scattered samples?

        Parameters
        ----------
        samples : array
            Array of positions, with coordinates along first axes.

        weights : array, default=None
            Sample weights, defaults to 1.

        dims : list
            List of labels (one for each mesh dimension).

        bins : int, default=30
            Number of bins i.e. mesh nodes to use along each dimension.

        method : string
            Method to interpolate (weighted) samples on mesh, either:

            - 'gaussian_kde': Gaussian kernel density estimation, based on :class:`scipy.stats.gaussian_kde`
            - 'cic' : Cloud-in-Cell assignment
            - 'histo' : simple binning.

        bw_method : string, default='scott'
            If ``method`` is ``'gaussian_kde'``, method to determine KDE bandwidth, see :class:`scipy.stats.gaussian_kde`.

        Returns
        -------
        mesh : Mesh
        """
        if not isinstance(samples,(tuple,list)):
            samples = [samples]
            dims = [dims]
        edges = []
        if not isinstance(bins,list): bins = [bins]*len(samples)
        if weights is None: weights = np.ones_like(samples[0])
        for d,b in zip(samples,bins):
            tmp = np.linspace(d.min(),d.max(),b+1) if np.ndim(b) == 0 else b
            edges.append(tmp)
        nodes = [(edge[:-1] + edge[1:])/2. for edge in edges]

        self = cls(mesh=None, dims=dims, edges=edges, nodes=nodes)
        if method == 'gaussian_kde':
            from scipy import stats
            density = stats.gaussian_kde(samples,weights=weights,bw_method=bw_method)
            nodes = self.mnodes()
            self.mesh = density(np.array([p.ravel() for p in nodes])).reshape(nodes[0].shape)
        elif method == 'cic':
            self.mesh = sample_cic(self.nodes,samples,weights=weights)/weights.sum()/self.volume()
        else:
            self.mesh = np.histogramdd(samples,weights=weights,bins=self.edges)[0]/weights.sum()/self.volume()
        return self

    @getaxes
    def __call__(self, axes=None):
        """
        Return mesh reduced to axes ``axes`` or dimensions ``dims``.
        Mesh is integrated along other dimensions.
        """
        isscalar = np.ndim(axes) == 0
        if isscalar: axes = [axes]
        intaxes = tuple([ax for ax in range(self.ndim) if ax not in axes])
        toret = self.integrate(axes=intaxes)
        x = [self.nodes[ax] for ax in axes]
        if toret.ndim <= 1: x = x[0]
        saxes = sorted(axes)
        return x,np.moveaxis(toret,range(len(axes)),[saxes.index(ax) for ax in axes])

    @getaxes_squeeze
    def argmax(self, axes=None):
        """Return coordinates of maximum along axes ``axes`` or dimensions ``dims``."""
        nodes,mesh = self(axes=axes)
        argmax = np.unravel_index(np.argmax(mesh),shape=mesh.shape)
        return np.array([node[arg] for node,arg in zip(nodes,argmax)])

    @getaxes_squeeze
    def argmean(self, axes=None):
        """Return mean coordinates along axes ``axes`` or dimensions ``dims``."""
        nodes,mesh = self(axes=axes)
        mnodes = np.meshgrid(*nodes,indexing='ij')
        average = [np.average(mnode,weights=mesh) for mnode in mnodes]
        return np.array(average)

    @getaxes
    def integrate(self, axes=None):
        """Integrate mesh along axes ``axes`` or dimensions ``dims``."""
        return (self.mesh*self.volume(axes=axes)).sum(axis=axes)

    def max(self):
        """Return mesh maximum value."""
        return self.mesh.max()

    def interpolate(self, points, method='linear'):
        """
        Interpolate mesh at provided points.

        Parameters
        ----------
        points : array
            Array of points, with coordinates along last axis.

        method : string
            Interpolation method, 'linear' or 'nearest'
            (see :func:`scipy.interpolate.interpn`)

        Returns
        -------
        values : array
            Mesh values at interpolated points.
        """
        return interpolate.interpn(self.nodes,self.mesh,points,method=method,bounds_error=False,fill_value=0.)

    @savefile
    def save_txt(self, filename, header='', fmt='%.8e', delimiter=' ', **kwargs):
        """
        Save mesh as ASCII file.
        TODO: needs major improvement, e.g. what about dimension names?

        Parameters
        ----------
        filename : string
            File name where to save mesh.

        kwargs : dict
            Arguments for :func:`numpy.savetxt`.
        """
        #print self.dims, self.nodes
        toret = [p.ravel() for p in self.mnodes()]
        toret.append(self.mesh.ravel()/self.max())
        if self.is_mpi_root():
            np.savetxt(filename,np.array(toret).T,header=header,fmt=fmt,delimiter=delimiter,**kwargs)

    def get_levels(self, targets):
        """Return density threshold(s) corresponding to confidence level(s) ``targets``."""
        integral = self.mesh*self.volume()

        def objective(x, target):
            return integral[self.mesh > x].sum() - target

        from scipy import optimize
        def _get_levels(target):
            try:
                return optimize.bisect(objective,0,self.mesh.max(),args=(target,))
            except ValueError:
                self.log_warning('bisection did not converge for confidence level {:.3f}.'.format(target))

        if np.ndim(targets) == 0:
            return _get_levels(targets)
        return [_get_levels(target) for target in targets]

    def get_sigmas(self, sigmas):
        """
        Return density thresholds corresponding to n-sigmas confidence levels ``sigmas``.

        Parameters
        ----------
        sigmas : int, list, array
            Sigmas to get density thresholds for.
            If scalar, return thresholds up to (including) ``sigmas``.

        Returns
        -------
        levels : array
            Density thresholds.
        """
        if np.ndim(sigmas) == 0: sigmas = 1 + np.arange(sigmas)
        targets = nsigmas_to_quantiles_1d(sigmas)
        return self.get_levels(targets)

    def __getstate__(self):
        """Return this class state dictionary."""
        state = {}
        for key in ['mesh','edges','nodes','dims']:
            if hasattr(self,key): state[key] = self.get(key)
        return state

    def deepcopy(self):
        """Return deep copy."""
        new = self.__new__(self.__class__)
        for key in ['mesh','edges','nodes','dims']:
            if hasattr(self,key): setattr(self,key,getattr(self,key).copy())
        return new

    def __radd__(self, other):
        """Operation corresponding to ``other + self``."""
        if other == 0: return self.deepcopy()
        return self.__add__(other)

    def __iadd__(self, other):
        """Operation corresponding to ``self += other``."""
        self.mesh += other.mesh
        return self

    def __add__(self, other):
        """Addition of two meshes."""
        new = self.deepcopy()
        new.mesh += other.mesh
        return new

    def __div__(self, other):
        """Division of two meshes."""
        new = self.deepcopy()
        new.mesh /= other
        return new
