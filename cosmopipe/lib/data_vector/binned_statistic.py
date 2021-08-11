"""Definition of :class:`BinnedStatistic` and :class:`BinnedProjection` to store binned data."""

import logging
import json
import re

import numpy as np

from cosmopipe.lib.utils import BaseClass, savefile
from cosmopipe.lib import utils
from .projection import ProjectionName


def _apply_matrices(array, matrices, axes=None):
    """
    Successively apply 2D matrix to ``array``.

    Parameters
    ----------
    array : ND array
        Array to apply matrices to.

    matrices : list of N 2D matrices
        i-th matrix is applied to ``axes[i]`` of ``array``

    axes : list, tuple
        ``array`` axes; defaults to ``range(len(matrices))``

    Returns
    -------
    array : ND array
    """
    if axes is None:
        axes = range(len(matrices))
    for ax,matrix in zip(axes,matrices):
        array = np.tensordot(array,matrix,axes=([ax],[-1])) # rescaled axis is at the end
        array = np.moveaxis(array,-1,ax)
    return array


def _matrix_new_edges(old_edges, edges, dims=None, flatten=False):
    """
    Return matrices to be applied to interpolate data binned with ``old_edges`` into ``edges``.
    Perform linear interpolation if ``edges`` are not a simple concatenation of ``old_edges``.

    Parameters
    ----------
    old_edges : dict, list
        Current dim: edges mapping. If list, should contain edges for each ``dim`` of ``dims``.

    edges : dict, list
        New dim: edges mapping. If list, should contain edges for each ``dim`` of ``dims``.

    dims : list, default=None
        Dimensions. Defaults to ``range(len(edges))``.

    flatten : bool, default=False
        Return single matrix to operate on the flattened data?

    Returns
    -------
    matrices : list of 2D arrays or 2D array
        List of matrices to be applied on data, or matrix to be applied on flattened data (if ``flatten`` is ``True``)

    new_edges : dict, list
        Dictionary of edges, if input ``edges`` is dictionary, else list of edges

    dims : list
        If input ``edges`` is dictionary, list of ``dims``.
    """
    if not old_edges:
        raise ValueError('Set current edges before updating them')
    if dims is None:
        dims = range(len(edges))
    if not isinstance(old_edges,dict):
        old_edges = dict(zip(dims,old_edges))
    new_edges = old_edges.copy()
    islist = not isinstance(edges,dict)
    if islist:
        edges = dict(zip(dims,edges))
    new_edges.update(edges)

    def get_matrix(dim):
        nedges = new_edges[dim]
        edges = old_edges[dim]
        frac_edges = np.interp(nedges,edges,np.arange(len(edges)))
        matrix = np.zeros((len(nedges)-1,len(edges)-1),dtype='f8')
        index_ = np.int32(frac_edges)
        index_low = index_[:-1]
        index_high = index_[1:]
        coeffs_low = (1. + np.floor(frac_edges) - frac_edges)[:-1]
        coeffs_high = (frac_edges - np.floor(frac_edges))[1:]
        for i,(ilow,iup) in enumerate(zip(index_low,index_high)):
            matrix[i,ilow:iup] = 1.
            matrix[i,ilow] = coeffs_low[i]
            if coeffs_high[0] != 0: # only when neceassary, as iup is not garanteed to be <= matrix.shape[1]
                matrix[i,iup] = coeffs_high[i]
        return matrix

    matrices = [get_matrix(dim) for dim in dims]

    if flatten:
        full_matrix = matrices[0]
        for matrix in matrices[1:]:
            tmp = full_matrix[:,None,:,None] * matrix[None,:,None,:]
            full_matrix = tmp.reshape((-1,full_matrix.shape[-1]*matrix.shape[-1]))
        matrices = full_matrix

    if islist:
        return matrices, [new_edges[dim] for dim in dims]
    return matrices, new_edges, dims


def _mask_edges(edges, mask):
    """Apply x-coordinate mask to of x-edges (array of length ``x.size + 1``)"""
    # apply mask on 2D array (edges[i], edges[i+1])
    edge = np.vstack([edges[:-1],edges[1:]]).T[mask]
    # keep the last edge
    return np.append(edge[:,0],edge[-1,1])


class RegisteredBinnedStatistic(type):

    """Metaclass registering :class:`BinnedStatistic` derived classes."""

    _registry = {}

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)
        meta._registry['.'.join((class_dict['__module__'],name))] = cls
        return cls


class BinnedStatistic(BaseClass,metaclass=RegisteredBinnedStatistic):
    """
    Class representing a binned statistic, similar to https://github.com/bccp/nbodykit/blob/master/nbodykit/binned_statistic.py.

    Attributes
    ----------
    data : dict
        Dictionary of data arrays, of same shape.

    edges : dict
        Dictionary of edges.

    dims : list
        List of dimension names.

    attrs : dict
        Dictionary of other attributes.
    """
    logger = logging.getLogger('BinnedStatistic')

    _title_template = '### {} ###'
    _default_mapping_header = {'shape':'.*?#shape = (.*)$','dims':'.*?#dims = (.*)$','edges':'.*?#edges (.*) = (.*)$','columns':'.*?#columns = (.*)$'}

    def __init__(self, data=None, edges=None, dims=None, attrs=None):
        """
        Initialize :class:`BinnedStatistic`.

        Parameters
        ----------
        data : dict, default=None
            Dictionary of data arrays, of same shape. Defaults to empty dictionary.

        edges : dict, list, default=None
            Dictionary of edges, or list of edges corresponding to ``dims``.
            If ``None``, no edges considered.

        dims : list, default=None
            List of dimension names. If ``None``, defaults to ``edges`` dictionary keys.

        attrs : dict, default=None
            Dictionary of other attributes.
        """
        if isinstance(data,self.__class__):
            self.__dict__.update(data.__dict__)
            return
        data = data or {}
        self.data = {str(col): np.asarray(value) for col,value in data.items()}
        edges = edges or {}
        if isinstance(edges,list):
            dims = dims or list(range(len(edges)))
            edges = {str(dim): edge for dim,edge in zip(dims,edges)}
        if dims is None:
            dims = list(edges.keys())
        self.edges = {str(dim): np.asarray(edges[dim]) for dim in dims if dim in edges}
        self.dims = dims
        if dims is None and not self.has_edges():
            self.dims = [str(i) for i in range(self.ndim)]
        self.attrs = attrs or {}

    def has_edges(self):
        """Has specified edges?"""
        return bool(self.edges)

    @property
    def columns(self):
        """Entries in :attr:`data`."""
        return list(self.data.keys())

    @property
    def ndim(self):
        """Number of dimensions of binned data."""
        return len(self.shape)

    @property
    def shape(self):
        """Shape of binned data; if :attr:`edges`, return tuple of length of edges - 1, else shape of first array in :attr:`data`"""
        if self.has_edges():
            return tuple(len(self.edges[dim])-1 for dim in self.dims)
        return self.data[self.columns[0]].shape

    @property
    def size(self):
        """Total size of binned data, i.e. product of length over all dimensions."""
        return np.prod(self.shape)

    def __getitem__(self, name):
        """If ``name`` is string, return corresponding array in :attr:`data`, else return copy sliced to ``name``."""
        if isinstance(name,str):
            return self.data[name]
        new = self.copy()
        new.data = {key: self.data[key][name] for key in self.data}
        if self.ndim == 1:
            name = (name,)
        if self.has_edges(): self.edges = {dim: _mask_edges(self.edges[dim],mask) for dim,mask in zip(self.dims,name)}
        return new

    def __setitem__(self, name, item):
        """If ``name`` is string, assign ``item`` to ``name`` entry of :attr:`data`, else assign ``item`` to slicing ``name``."""
        if isinstance(name,str):
            self.data[name] = item
        else:
            for key in self.data:
                self.data[key][name] = item

    def __getstate__(self):
        """Return this class state dictionary."""
        state = {}
        for key in ['data','edges','dims','attrs']:
            state[key] = getattr(self,key)
        state['__class__'] = '.'.join((self.__class__.__module__,self.__class__.__name__))
        return state

    def _matrix_new_edges(self, edges, dims=None, flatten=False, weights=None):
        """Return matrix/matrices to fit binned data into new ``edges``."""
        if dims is None: dims = self.dims
        if isinstance(edges,list): edges = dict(zip(dims,edges))
        matrix, edges, dims = _matrix_new_edges(self.edges,edges,dims,flatten=flatten)
        if weights is None and flatten:
            weights = self.attrs.get('weights',None)
        if weights is not None:
            if not flatten:
                raise ValueError('Cannot apply weights if not flattened')
            if isinstance(weights,str):
                weights = self[weights]
            weights = weights.flatten()
            matrix = matrix * weights[None,:]
        return matrix, edges, dims

    def set_new_edges(self, edges, dims=None, weights=None, columns_to_sum=None):
        """
        Interpolate binned data within new edges. Perform linear interpolation if ``edges`` are not a simple concatenation of current :attr:`edges`.

        Parameters
        ----------
        edges : dict, list
            New dim: edges mapping. If list, should contain edges for each ``dim`` of ``dims``.

        dims : list, default=None
            List of dimension names. Defaults to :attr:`dims`.

        weights : array, default=None
            Array of weights (of shape :attr:`shape`). If ``None``, defaults to 1.

        columns_to_sum: list
            List of columns to sum, i.e. to not renormalize after rebinning.

        Warning
        -------
        Requires :attr:`edges` to be set.
        """
        if columns_to_sum is None: columns_to_sum = self.attrs.get('columns_to_sum',[])
        if weights is None:
            weights = self.attrs.get('weights',None)

        if weights is not None:
            if isinstance(weights,str):
                columns_to_sum.append(weights)
                weights = self[weights]
        else:
            weights = np.ones(self.shape,dtype='f8')

        if np.ndim(dims) == 0:
            edges = [edges]
            dims = [dims]
        matrices, self.edges, dims = self._matrix_new_edges(edges,dims=dims,flatten=False,weights=None)

        axes = [self.dims.index(dim) for dim in dims]
        binned_weights = _apply_matrices(weights,matrices,axes=axes)
        for col in self.data:
            self.data[col] = _apply_matrices(self.data[col]*weights,matrices,axes=axes)
            if col not in columns_to_sum:
                self.data[col] /= binned_weights

    def _matrix_rebin(self, factors, dims=None, flatten=False, weights=None):
        """Return matrix/matrices to rebin data by factor ``factors``."""
        if dims is None: dims = self.dims
        if isinstance(factors,list): factors = dict(zip(dims,factors))
        if weights is None:
            weights = self.attrs.get('weights',None)
        edges = {}
        old_edges = {dim: np.arange(size+1) for dim,size in zip(self.dims,self.shape)}
        for dim,factor in factors.items():
            old_edge = old_edges[dim]
            if len(old_edge) - 1 % factor != 0:
                raise ValueError('Input rebinning factor {:d} does not divide data size {:d} along {}'.format(factor,len(old_edge)-1,dim))
            edges[dim] = old_edge[::factor]
        return _matrix_new_edges(old_edges,edges,dims=dims,flatten=flatten,weights=weights)[::2]

    def rebin(self, factors, dims=None, weights=None, columns_to_sum=None):
        """
        Rebin data by factor ``factors``.

        Parameters
        ----------
        factors : dict, list
            dim: rebinning factor mapping. If list, should contain rebinning factor for each ``dim`` of ``dims``.

        dims : list, default=None
            List of dimension names. Defaults to :attr:`dims`.

        weights : array, default=None
            Array of weights (of shape :attr:`shape`). If ``None``, defaults to 1.

        columns_to_sum: list
            List of columns to sum, i.e. to not renormalize after rebinning.
        """
        if weights is None:
            weights = self.attrs.get('weights',None)
        columns_to_sum = columns_to_sum or []

        if weights is not None:
            if isinstance(weights,str):
                columns_to_sum.append(weights)
                weights = self[weights]
        else:
            weights = np.ones(self.shape,dtype='f8')

        matrices, dims = self._matrix_rebin(factors,dims=dims,flatten=False,weights=None)

        axes = [self.dims.index(dim) for dim in dims]
        binned_weights = _apply_matrices(weights,matrices,axes=axes)
        for col in self.data:
            self.data[col] = _apply_matrices(self.data[col]*weights,matrices,axes=axes)
            if col not in columns_to_sum:
                self.data[col] /= binned_weights

    def squeeze(self, dims=None):
        """
        Squeeze binned data along dimensions ``dims``.
        If ``dims`` is ``None``, defaults to dimensions with length <= 1.
        :attr:`dims` and :attr:`edges` are updated to match new shape.
        """
        if dims is None:
            dims = [dim for idim,dim in enumerate(self.dims) if self.shape[idim] <= 1]
        if np.ndim(dims) == 0: dims = [dims]
        axes = tuple([self.dims.index(dim) for dim in dims])
        for col in self.data:
            self.data[col] = np.squeeze(self.data[col],axis=axes)
        for dim in dims:
            del self.dims[self.dims.index(dim)]
            if dim in self.edges:
                del self.edges[dim]

    def average(self, dims=None, weights=None, columns_to_sum=None):
        """
        Average binned data along dimensions ``dims``.
        This is equivalent to :meth:`rebin` with factors corresponding to lengths along dimensions ``dims``,
        followed by :meth:`squeeze` along those dimensions.
        """
        if dims is None:
            dims = self.dims
        if np.ndim(dims) == 0:
            dims = [dims]
        axes = [self.dims.index(dim) for dim in dims]
        factors = {dim: shape[axis] for dim,axis in zip(dims,axes)}
        self.rebin(factors,weights=weights,columns_to_sum=columns_to_sum)
        self.squeeze(dims=dims)

    @savefile
    def save_txt(self, filename=None, fmt='.18e', comments='#', ignore_json_errors=True):
        """
        Dump :class:`BinnedStatistic`.

        Parameters
        ----------
        filename : string, default=None
            ASCII file name where to save binned data.
            If ``None``, do not write on disk.

        fmt : string, default='.18e'
            Floating point format.

        comments : string, default='#'
            String that will be prepended to the header lines.

        ignore_json_errors : bool, default=True
            When trying to dump :attr:`attrs` using *json*, ignore errors.

        Returns
        -------
        lines : list
            List of strings (lines).
        """
        lines = self.get_header_txt(comments=comments,ignore_json_errors=ignore_json_errors)
        data = {col:self.data[col].flatten() for col in self.columns}
        for ii in range(self.size):
            lines.append(' '.join(['{:{fmt}}'.format(data[col][ii],fmt=fmt) for col in data]))
        if filename is not None:
            if self.is_mpi_root():
                with open(filename,'w') as file:
                    for line in lines:
                        file.write(line + '\n')
        return lines

    def get_header_txt(self, comments='#', ignore_json_errors=True):
        """
        Dump header:

        - items in :attr:`attrs`
        - :attr:`shape`
        - :attr:`dims`
        - :attr:`edges`
        - :attr:`columns`

        Parameters
        ----------
        comments : string, default='#'
            String to be prepended to the header lines.

        ignore_json_errors : bool, default=True
            When trying to dump :attr:`attrs` using *json*, ignore errors.

        Returns
        -------
        header : list
            List of strings (lines).
        """
        header = ['{}{}'.format(comments,self.get_title_label())]
        for key,value in self.attrs.items():
            try:
                header.append('{}{} = {}'.format(comments,key,json.dumps(value)))
            except TypeError:
                if not ignore_json_errors:
                    raise
                if isinstance(value,np.ndarray):
                    value = value.tolist()
                header.append('{}{} = {}'.format(comments,key,value))
        header.append('{}#shape = {}'.format(comments,json.dumps(self.shape)))
        header.append('{}#dims = {}'.format(comments,json.dumps(self.dims)))
        if self.has_edges():
            for dim in self.edges:
                header.append('{}#edges {} = {}'.format(comments,dim,json.dumps(self.edges[dim].tolist())))
        header.append('{}#columns = {}'.format(comments,json.dumps(self.columns)))
        return header

    @classmethod
    def load_txt(cls, filename, comments='#', usecols=None, skip_rows=0, max_rows=None, mapping_header=None, pattern_header=None, attrs=None, **kwargs):
        """
        Load :class:`BinnedStatistic` from disk.

        Note
        ----
        If previously saved using :meth:`save_txt`, loading the :class:`BinnedStatistic` only requires ``filename``.
        In this case, the returned instance will be of the class that was used to create it (e.g. :class:`BinnedProjection` below)
        - not necessarily :class:`BinnedStatistic`.

        Parameters
        ----------
        filename : string
            File name to read in.

        comments : string, default='#'
            Characters used to indicate the start of a comment.

        usecols : list, default=None
            Which columns to read, with 0 being the first.

        skip_rows : int, default=0
            Skip the first ``skip_rows`` lines, including comments.

        max_rows : int, default=None
            Read ``max_rows lines`` of content after ``skip_rows`` lines. The default is to read all the lines.

        mapping_header : dict, default=None
            Dictionary holding key:regex mapping or (regex, type) to provide the type.
            The corresponding values, read in the header, will be saved in the :attr:`attrs` dictionary.

        pattern_header : string, default=None
            A regex pattern for header with groups corresponding to key, value to add into the :attr:`attrs` dictionary.

        attrs : dict, default=None
            Attributes to save in the :attr:`attrs` dictionary.

        kwargs : dict
            Arguments for :meth:`__init__` (other than ``data`` and ``attrs``).

        Returns
        -------
        data : BinnedStatistic
        """
        def get_file(file_):
            file = []
            for iline,line in enumerate(file_):
                if max_rows is not None and iline >= skip_rows + max_rows:
                    break
                if iline >= skip_rows:
                    file.append(line)
            return file

        if isinstance(filename,str):
            cls.log_info('Loading {}.'.format(filename),rank=0)
            with open(filename,'r') as file_:
                file = get_file(file_)
        else:
            file = get_file(filename)

        file = file[skip_rows:]
        tmpcls = cls.read_title_label(file[0][len(comments):])
        self_format = tmpcls is not None
        if self_format: cls = tmpcls
        if pattern_header is None and self_format:
            pattern_header = '(.*) = (.*)$'

        header = cls.read_header_txt(file,comments=comments,mapping_header=mapping_header,pattern_header=pattern_header)
        attrs = (attrs or {}).copy()
        for name,value in header.items():
            if name in cls._default_mapping_header and kwargs.get(name,None) is None:
                kwargs[name] = value
            elif attrs.get(name,None) is None:
                attrs[name] = value
        shape, columns = kwargs.pop('shape',None), kwargs.pop('columns',None)

        def str_to_float(e):
            return float(e)

        data = {}
        if columns is not None:
            data = {col:[] for col in columns}

        for iline,line in enumerate(file):
            if line.startswith(comments): continue
            line = line.strip().split()
            if usecols is None:
                usecols = range(len(line))
            if columns is None:
                columns = ['col{:d}'.format(icol) for icol in usecols]
                data = {col:[] for col in columns}
            for icol,col in zip(usecols,columns):
                data[col].append(str_to_float(line[icol]))
        for col in columns:
            data[col] = np.array(data[col])
            if shape is not None:
                data[col] = data[col].reshape(shape)

        return cls(data=data,attrs=attrs,**kwargs)

    def __copy__(self):
        """Return shallow copy, including that of :attr:`data` internal dictionary, :attr:`dims`, :attr:`edges` and attr:`attrs`."""
        new = super(BinnedStatistic,self).__copy__()
        for name in ['data','dims','edges','attrs']:
            setattr(new,name,getattr(new,name).copy())
        return new


@classmethod
def get_title_label(cls):
    """Return title, as ``module.class``."""
    return cls._title_template.format('.'.join((cls.__module__,cls.__name__)))


@classmethod
def read_title_label(cls, line):
    """
    Decode title ``line``, splitting ``module.class`` into (``module``, ``class``).
    It loads ``module``, then if ``class`` is in :attr:`_registry`, return corresponding class.
    Else return ``None``.
    """
    template = cls._title_template.replace('{}','(.*)')
    match = re.match(template,line)
    if match:
        module_class = match.group(1)
        import importlib
        importlib.import_module(re.match('(.*)\.(.*)$',module_class).group(1))
        if module_class in cls._registry:
            return cls._registry[module_class]


@classmethod
def read_header_txt(cls, file, comments='#', mapping_header=None, pattern_header=None, ignore_json_errors=True):
    """
    Read and decode header.

    Parameters
    ----------
    file : list, iterator
        List of lines.

    comments : string, default='#'
        Characters used to indicate the start of a header line.

    mapping_header : dict, default=None
        Dictionary holding key:regex mapping or (regex, type) to provide the type.
        Type can be unspecified (or ``None``), in which case decoded will be tried with *json*,
        a string corresponding to ``__builtins__``, or a callable.

    pattern_header : string, default=None
        A regex pattern with groups corresponding to key:value.

    ignore_json_errors : bool, default=True
        When trying to decode header values using *json*, ignore errors.

    Returns
    -------
    attrs : dict
    """
    attrs = {}
    mapping_header = (mapping_header or {}).copy()
    mapping_header = utils.dict_nonedefault(mapping_header,**cls._default_mapping_header)

    def decode_value(value, decode):
        # if None try json, if string, builtins, else call
        if decode is None:
            try:
                value = json.loads(value)
            except json.decoder.JSONDecodeError:
                if not ignore_json_errors:
                    raise
        elif isinstance(decode,str):
            value = __builtins__[decode](value)
        else:
            value = decode(value)
        return value

    def fill_attrs_values(decode, key, *values):
        if len(values) > 1:
            name,value = values
            if key not in attrs:
                attrs[key] = {}
            attrs[key][name] = decode_value(value,decode)
        else:
            attrs[key] = decode_value(values[0],decode)

    for line in file:
        if not line.startswith(comments):
            break # header is assumed contiguous, here we fall on an uncommented line, so we break
        match = None
        for key,pattern in mapping_header.items():
            if isinstance(pattern,tuple):
                pattern,decode = pattern
            else:
                decode = None # no rule to decode
            match = re.match(pattern,line[len(comments):])
            if match is not None:
                fill_attrs_values(decode,key,*match.groups())
                cls.log_debug('Setting attribute {} = {} from header.'.format(key,attrs[key]),rank=0)
                break
        if not match:
            if pattern_header:
                match = re.match(pattern_header,line[len(comments):])
                if match is not None:
                    fill_attrs_values(decode,*match.groups())

    for key in mapping_header.items():
        if key not in attrs:
            cls.log_debug('Could not find attribute {} in header'.format(key))

    return attrs


BinnedStatistic.get_title_label = get_title_label
BinnedStatistic.read_title_label = read_title_label
BinnedStatistic.read_header_txt = read_header_txt


class BinnedProjection(BinnedStatistic):
    """
    Class representing a binned projection, i.e. a :class:`BinnedStatistic` with a :class:`ProjectionName` attribute.
    Can be, e.g., the power spectrum monopole. Dimensions are x-coordinates.

    Attributes
    ----------
    proj : ProjectionName
        Projection.
    """
    logger = logging.getLogger('BinnedProjection')
    _default_mapping_header = {**BinnedStatistic._default_mapping_header,'proj':'.*?#proj = (.*)$','y':'.*?y = (.*)$'}

    def __init__(self, data=None, x=None, y=None, edges=None, dims=None, weights=None, proj=None, attrs=None):
        """
        Initialize :class:`BinnedProjection`.

        Parameters
        ----------
        data : dict, default=None
            Dictionary of data arrays, of same shape. Defaults to empty dictionary.

        x : tuple, string, array, default=None
            Name of x-coordinate(s) in ``data``, or array (e.g. ``'k'``, ``('s','mu')``)

        y : string, array, default=None
            Name of y-coordinate in ``data``, or array.

        edges : dict, list, default=None
            Dictionary of edges, or list of edges corresponding to ``dims``.
            If ``None``, no edges considered.

        dims : list, default=None
            List of dimension names. If ``None``, defaults to ``edges`` dictionary keys.

        weights : string, array, default=None
            Name of weights in ``data``, or array. These will be used to rebin data;
            e.g. number of modes for the power spectrum, RR pair counts for the correlation function.

        proj : ProjectionName, string, tuple, dict, default=None
            Projection.

        attrs : dict, default=None
            Dictionary of other attributes.
        """
        data = data or {}
        if isinstance(x,str):
            x = [x,]
        if x is not None and not isinstance(x[0],str):
            #if x is None: x = np.arange(len(y))
            data['x'] = np.asarray(x)
            dims = x = ['x',]
        if y is not None and not isinstance(y,str):
            data['y'] = np.asarray(y)
            y = 'y'
        if weights is not None and not isinstance(weights,str):
            data['weights'] = np.asarray(weights)
            weights = 'weights'
        super(BinnedProjection,self).__init__(data=data,dims=dims,edges=edges,attrs=attrs)
        for label,column in zip(['y','weights'],[y,weights]):
            if column is not None:
                self.attrs[label] = column
        if x is None:
            if dims is None and edges is None: # self.dims set if dims or edges provided
                self.dims = ['x']
        else:
            self.dims = list(x)
        self.proj = ProjectionName(proj)

    def has_x(self):
        """Are x-coordinates specified?"""
        return all(x in self.data for x in self.dims)

    def has_y(self):
        """Are y-coordinates specified?"""
        return 'y' in self.attrs and self.attrs['y'] in self.data

    def get_x_average(self, xlim=None, mask=Ellipsis, weights=None, from_edges=None):
        """
        Return average of x-coordinates. e.g., if x is ``(s, mu)``,
        the average of ``s`` over ``mu`` and the average of ``mu`` over ``s``.

        Parameters
        ----------
        xlim : list, tuple, default=None
            x-limits for each x-coordinate.

        mask : list, tuple, default=Ellipsis
            1D mask for each x-coordinate.

        weights : array, default=None
            Array of same shape as data.

        from_edges : bool, default=None
            If ``True``, return points at mid-edges.
            If ``None``, if x-coordinates are not in data, return mid-points.
            Else, average of x-coordinates along each dimensions are returned.

        Returns
        -------
        x : tuple of 1D arrays
        """
        if from_edges is None: from_edges = not self.has_x()
        masks = self.get_index(xlim=xlim,mask=mask,flatten=False)

        def mid(edges):
            return (edges[:-1] + edges[1:])/2.

        if from_edges:
            return [mid(self.edges[dim])[masks[idim]] for idim,dim in enumerate(self.dims)]

        x = self.get_x(flatten=False)
        if np.ndim(x) == 1:
            x = x[:,None]
        allaxes = list(range(self.ndim))
        toret = []
        for idim,dim in enumerate(self.dims):
            axes = allaxes.copy()
            del axes[idim]
            toret.append(np.average(x[...,idim],axis=tuple(axes),weights=weights))
        #if self.ndim == 1:
        #    return toret[0]
        return tuple(toret)

    def get_x(self, xlim=None, mask=Ellipsis, flatten=True):
        """
        Return x-coordinates within ``xlim`` or ``mask``.

        Parameters
        ----------
        xlim : list, tuple, default=None
            x-limits for each x-coordinate.

        mask : list, tuple, default=Ellipsis
            Mask for each x-coordinate.
            If ``flatten`` is ``False``, must be of same length as data along each x-coordinate,
            else same size (i.e. product of shape) as data.

        flatten : bool, default=True
            If ``True``, return flattened x.
            Else, :attr:`ndim`-D x.

        Returns
        -------
        x : array
            (masked) x-coordinates (1D if flatten, else :attr:`ndim`-D), stacked along last axis.
            If only one x-coordinate, last dimension is removed.
        """
        x = [self[x] for x in self.dims]
        if xlim is None and mask is Ellipsis:
            index = Ellipsis if flatten else [Ellipsis]*self.ndim
        else:
            index = self.get_index(xlim,mask=mask,flatten=flatten)
        if flatten:
            x = np.moveaxis([x_.flatten()[index] for x_ in x],0,-1)
            if x.shape[-1] == 1:
                return x[...,0]
            return x
        x = np.moveaxis([np.take(x_,index[axis],axis=axis) if index[axis] is not Ellipsis else x_ for axis,x_ in enumerate(x)],0,-1)
        if x.shape[-1] == 1:
            return x[...,0]
        return x

    def get_y(self, xlim=None, mask=Ellipsis, flatten=True):
        """Same as :meth:`get_x`, for y-coordinate."""
        y = self[self.attrs['y']]
        if flatten:
            return y.flatten()[self.get_index(xlim,mask=mask,flatten=flatten)]
        return y[self.get_index(xlim,mask=mask,flatten=flatten)]

    def get_edges(self, xlim=None, mask=Ellipsis):
        """
        Return x-edges within ``xlim`` or ``mask``.

        Parameters
        ----------
        xlim : list, tuple, default=None
            x-limits for each x-coordinate.

        mask : list, tuple, default=Ellipsis
            Mask for each x-coordinate, of same length as data along each x-coordinate.

        Returns
        -------
        edges : tuple
            (masked) edges along each x-coordinate.
        """
        # mask only in the form (1d, 1d, ...)
        masks = self.get_index(xlim=xlim,mask=mask,flatten=False)
        return tuple(_mask_edges(self.edges[dim],mask_) for dim,mask_ in zip(self.dims,masks))

    def set_x(self, x, mask=Ellipsis, flatten=True):
        """
        Set x-coordinates.

        Parameters
        ----------
        x : list, array
            New x-coordinates. Can be a single array if :attr:`ndim` is 1.
            If ``flatten`` is ``False``, arrays must be of same shape as (masked) data.
            Else, arrays must be 1D, of same size as (masked) data.

        mask : list, tuple, default=Ellipsis
            Mask for each x-coordinate.
            If ``flatten`` is ``False``, must be of same length as data along each x-coordinate,
            else same size (i.e. product of shape) as data.

        flatten : bool, default=True
            Whether input is flattened.
        """
        if flatten:
            for ix,x_ in enumerate(self.dims):
                self.data.setdefault(x_,np.full(self.shape,np.nan))
                self[x_].flat[mask] = x[...,ix] if x.ndim > 1 else x
        else:
            if mask is Ellipsis or np.ndim(mask) == 1:
                mask = [mask]*self.ndim
            mask = np.ix_(*mask)
            for ix,x_ in enumerate(self.dims):
                self.data.setdefault(x_,np.full(self.shape,np.nan))
                self[x_][mask] = x[...,ix] if x.ndim > 1 else x

    def set_y(self, y, mask=Ellipsis, flatten=True):
        """Same as :meth:`set_x`, for y-coordinate."""
        self.attrs.setdefault('y','y')
        self.data.setdefault(self.attrs['y'],np.full(self.shape,np.nan))
        y_ = self.attrs['y']
        if flatten:
            self[y_].flat[mask] = y
        else:
            if mask is Ellipsis or np.ndim(mask) == 1:
                mask = [mask]*self.ndim
            mask = np.ix_(*mask)
            self[y_][np.ix_(*mask)] = y

    def get_index(self, xlim=None, mask=Ellipsis, flatten=True):
        """
        Return index.

        Parameters
        ----------
        xlim : list, tuple, default=None
            x-limits for each x-coordinate.

        mask : list, tuple, default=Ellipsis
            Mask for each x-coordinate.
            If ``flatten`` is ``False``, must be of same length as data along each x-coordinate,
            else same size (i.e. product of shape) as data.

        flatten : bool, default=True
            If ``True``, return index in flatten data array.
            Else, return tuple of 1D index along each dimension.
        """
        if xlim is not None and np.ndim(xlim) == 1:
            xlim = [xlim]*self.ndim
        if flatten:
            mask_ = np.zeros(self.size,dtype='?')
            mask_[mask] = True
            if xlim is not None:
                x = self.get_x()
                if self.ndim == 1: x = x[:,None] # get_x() automatically squeezes ndim = 1
                for idim in range(self.ndim):
                    mask_ &= (x[...,idim] >= xlim[idim][0]) & (x[...,idim] <= xlim[idim][-1])
            return np.flatnonzero(mask_)
        toret = []
        x = self.get_x(flatten=False)
        allaxes = list(range(self.ndim))
        if mask is Ellipsis or np.ndim(mask) == 1:
            mask = [mask]*self.ndim
        for idim,dim in enumerate(self.dims):
            mask_ = np.zeros(self.shape[idim],dtype='?')
            mask_[mask[idim]] = True
            if xlim is not None:
                tmp = (x[...,idim] >= xlim[0]) & (x[...,idim] <= xlim[-1])
                axes = allaxes.copy()
                del axes[idim]
                mask_ &= np.all(tmp,axis=tuple(axes))
            toret.append(np.flatnonzero(mask_))
        return tuple(toret)

    def __getstate__(self):
        """Return this class state dictionary."""
        state = super(BinnedProjection,self).__getstate__()
        state['proj'] = self.proj.__getstate__()
        return state

    def __setstate__(self, state):
        """Set this class state dictionary."""
        super(BinnedProjection,self).__setstate__(state)
        self.proj = ProjectionName.from_state(self.proj)

    def get_header_txt(self, comments='#', **kwargs):
        """Return header, adding :attr:`proj` to that of :class:`BinnedStatistic`."""
        header = super(BinnedProjection,self).get_header_txt(comments=comments,**kwargs)
        header += ['{}#proj = {}'.format(comments,json.dumps(self.proj.__getstate__()))]
        return header
