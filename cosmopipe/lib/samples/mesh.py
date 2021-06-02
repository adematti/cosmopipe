import logging

import numpy as np

from cosmopipe.lib.utils import BaseClass, savefile
from cosmopipe.lib import mpi
from .utils import *


def sample_cic(edges, samples, weights=None):
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


class Mesh(BaseClass):

    logger = logging.getLogger('Mesh')

    @mpi.MPIInit
    def __init__(self, mesh, names, edges=None, nodes=None, isdensity=True):
        self.mesh = mesh
        self.names = names
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
        return tuple(len(edges)-1 for edges in self.edges)

    @property
    def ndim(self):
        return len(self.edges)

    def widths(self, axis=None):
        if axis is None: axis = range(self.ndim)
        elif np.isscalar(axis):
            return np.diff(self.edges[axis])
        return [self.widths(ax) for ax in axis]

    def ranges(self, axis=None):
        if axis is None: axis = range(self.ndim)
        elif np.isscalar(axis):
            return (self.edges[axis].min(),self.edges[axis].max())
        return [self.ranges(ax) for ax in axis]

    def mnodes(self, axis=None):
        if axis is None: axis = range(self.ndim)
        isscalar = np.isscalar(axis)
        if isscalar:
            axis = [axis]
        toret = np.meshgrid(*[self.nodes[ax] for ax in axis],indexing='ij')
        if isscalar:
            return toret[0]
        return toret

    def volume(self, axis=None):
        if axis is None: axis = range(self.ndim)
        if np.isscalar(axis): axis = [axis]
        return np.prod(np.meshgrid(*self.widths(axis),indexing='ij'),axis=0) # very memory inefficient...

    def get_axis(self, names=None):
        if names is None: names = self.names
        return [self.names.index(param) for param in names]

    @classmethod
    def from_samples(cls, samples, weights=None, names=None, bins=30, method='cic', bw_method='scott'):
        if not isinstance(samples,(tuple,list)):
            samples = [samples]
            names = [names]
        edges = []
        if not isinstance(bins,list): bins = [bins]*len(samples)
        if weights is None: weights = np.ones_like(samples[0])
        for d,b in zip(samples,bins):
            tmp = np.linspace(d.min(),d.max(),b) if np.ndim(b) == 0 else b
            edges.append(tmp)
        nodes = [(edge[:-1] + edge[1:])/2. for edge in edges]

        self = cls(mesh=None, names=names, edges=edges, nodes=nodes)
        #print('LOOOL',samples[0].shape,np.isnan(samples[0]).any(),np.isnan(samples[0]).sum(),np.isinf(samples[0]).any(),names,samples[0])
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

    def __call__(self, names):
        if np.isscalar(names):
            names = [names]
        ndim = len(names)
        indices = self.get_axis(names)
        axis = tuple([axis for axis in range(self.ndim) if axis not in indices])
        toret = self.integrate(axis=axis)
        x = [self.nodes[ax] for ax in indices]
        if ndim == 1: x = x[0]
        sindices = sorted(indices)
        return x,np.moveaxis(toret,range(len(indices)),[sindices.index(ind) for ind in indices])

    def maximum(self, names=None):
        if names is None:
            mesh,nodes = self.mesh,self.nodes
        else:
            nodes,mesh = self(names)
            if len(names) == 1: nodes = [nodes]
        argmax = np.unravel_index(np.argmax(mesh),shape=mesh.shape)
        return np.array([node[arg] for node,arg in zip(nodes,argmax)])

    def mean(self, names=None):
        if names is None:
            mesh,nodes = self.mesh,self.nodes
        else:
            nodes,mesh = self(names)
            if len(names) == 1: nodes = [nodes]
        mnodes = np.meshgrid(*nodes,indexing='ij')
        average = [np.average(mnode,weights=mesh) for mnode in mnodes]
        return np.array(average)

    def integrate(self, axis=None):
        return (self.mesh*self.volume(axis=axis)).sum(axis=axis)

    def max(self):
        return self.mesh.max()

    def interpolate(self, points, kind_interpol='linear'):
        interpolated_mesh = interpolate.interpn(self.nodes,self.mesh,kind=kind_interpol,bounds_error=False,fill_value=0.)
        return interpolated_mesh(*points)

    @savefile
    def save_txt(self, filename, header='', fmt='%.8e', delimiter=' ', **kwargs):
        #print self.names, self.nodes
        toret = [p.ravel() for p in self.mnodes()]
        toret.append(self.mesh.ravel()/self.max())
        if self.is_mpi_root():
            np.savetxt(filename,np.array(toret).T,header=header,fmt=fmt,delimiter=delimiter,**kwargs)

    def get_levels(self, targets):
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
        if np.ndim(sigmas) == 0: sigmas = 1 + np.arange(sigmas)
        targets = nsigmas_to_quantiles_1d(sigmas)
        return self.get_levels(targets)

    def __getstate__(self):
        state = {}
        for key in ['mesh','edges','nodes','names']:
            if hasattr(self,key): state[key] = self.get(key)
        return state

    def deepcopy(self):
        new = self.__new__(self.__class__)
        for key in ['mesh','edges','nodes','names']:
            if hasattr(self,key): setattr(self,key,getattr(self,key).copy())
        return new

    def __radd__(self, other):
        if other == 0: return self.deepcopy()
        return self.__add__(other)

    def __add__(self, other):
        new = self.deepcopy()
        new.mesh += other.mesh
        return new

    def __div__(self, other):
        new = self.deepcopy()
        new.mesh /= other
        return new
