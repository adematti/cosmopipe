import logging
import json
import re

import numpy as np

from cosmopipe.lib.utils import BaseClass, savefile
from cosmopipe.lib import utils
from .projection import ProjectionName


def _apply_matrices(array, matrices):
    for im,matrix in enumerate(matrices):
        array = np.tensordot(array,matrix,axes=([im],[-1])) # rescaled axis is at the end
        array = np.moveaxis(array,-1,im)
    return array


def _matrix_new_edges(old_edges, edges, dims=None, flatten=False):
    if not old_edges:
        raise ValueError('Set current edges before setting new ones')
    new_edges = old_edges.copy()
    if not isinstance(edges,dict):
        if np.ndim(dims) == 0:
            dims = [dims]
            edges = [edges]
        edges = {dim:edge for dim,edge in zip(dims,edges)}
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

    matrices = [get_matrix(dim) for dim in old_edges.keys()]

    if flatten:
        full_matrix = matrices[0]
        for matrix in matrices[1:]:
            tmp = full_matrix[:,None,:,None] * matrix[None,:,None,:]
            full_matrix = tmp.reshape((-1,full_matrix.shape[-1]*matrix.shape[-1]))
        return new_edges, full_matrix

    return matrices, new_edges


def _mask_edges(edges, masks):
    toret = {}
    for key,mask in zip(edges,masks):
        e = np.vstack([edges[key][:-1],edges[key][1:]]).T[mask]
        toret[key] = np.append(e[:,0],e[-1,1])
    return toret


_title_template = '### {} ###'


class RegisteredBinnedStatistic(type):

    _registry = {}

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)
        meta._registry['.'.join((class_dict['__module__'],name))] = cls
        return cls


class BinnedStatistic(BaseClass,metaclass=RegisteredBinnedStatistic):
    """
    Structure holding a binned statistic.
    """
    logger = logging.getLogger('BinnedStatistic')

    _default_mapping_header = {'shape':'.*?#shape = (.*)$','dims':'.*?#dims = (.*)$','edges':'.*?#edges (.*) = (.*)$','columns':'.*?#columns = (.*)$'}

    def __init__(self, data=None, edges=None, dims=None, attrs=None):
        if isinstance(data,self.__class__):
            self.__dict__.update(data.__dict__)
            return
        data = data or {}
        self.data = {col: np.asarray(value) for col,value in data.items()}
        edges = edges or {}
        if isinstance(edges,list):
            dims = dims or list(range(len(edges)))
            edges = {dim: edge for dim,edge in zip(dims,edges)}
        if dims is None:
            dims = list(edges.keys())
        self.edges = {dim:np.asarray(edges[dim]) for dim in dims if dim in edges}
        self.dims = dims
        if dims is None and not self.has_edges():
            self.dims = list(range(self.ndim))
        self.attrs = attrs or {}

    def has_edges(self):
        return bool(self.edges)

    @property
    def columns(self):
        return list(self.data.keys())

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def shape(self):
        if self.has_edges():
            return tuple(len(self.edges[dim])-1 for dim in self.dims)
        return self.data[self.columns[0]].shape

    @property
    def size(self):
        return np.prod(self.shape)

    def __getitem__(self, name):
        if isinstance(name,str):
            return self.data[name]
        new = self.copy()
        new.data = {key: self.data[key][name] for key in self.data}
        if self.ndim == 1:
            name = (name,)
        new.edges = _mask_edges(self.edges,name)
        return new

    def __setitem__(self, name, item):
        if isinstance(name,str):
            self.data[name] = item
        else:
            for key in self.data:
                self.data[key][name] = item

    def __getstate__(self):
        state = {}
        for key in ['data','edges','dims','attrs']:
            state[key] = getattr(self,key)
        state['__class__'] = '.'.join((self.__class__.__module__,self.__class__.__name__))
        return state

    def _matrix_new_edges(self, edges, dims=None, flatten=False, weights=None):
        matrix, edges = _matrix_new_edges(self.edges,edges,dims=dims,flatten=flatten)
        if weights is None and flatten:
            weights = self.attrs.get('weights',None)
        if weights is not None:
            if not flatten:
                raise ValueError('Cannot apply weights if not flattened')
            if isinstance(weights,str):
                weights = self[weights]
            weights = weights.flatten()
            binned_weights = matrix.dot(weights)
            matrix = matrix * binned_weights[:,None] * weights[None,:]
        return matrix, edges

    def set_new_edges(self, edges, dims=None, weights=None, columns_to_sum=None):
        columns_to_sum = columns_to_sum or self.attrs.get('columns_to_sum',[])
        if weights is None:
            weights = self.attrs.get('weights',None)

        if weights is not None:
            if isinstance(weights,str):
                columns_to_sum.append(weights)
                weights = self[weights]
        else:
            weights = np.ones(self.shape,dtype='f8')

        matrices, self.edges = self._matrix_new_edges(edges,dims=dims,flatten=False,weights=None)

        binned_weights = _apply_matrices(weights,matrices)
        for col in self.data:
            self.data[col] = _apply_matrices(self.data[col]*weights,matrices)
            if col not in columns_to_sum:
                self.data[col] /= binned_weights

    def _matrix_rebin(self, factors, dims=None, flatten=False, weights=None):
        if weights is None:
            weights = self.attrs.get('weights',None)
        if dim is not None:
            factors = {dim:factors}
        edges = {}
        old_edges = {dim: np.arange(size+1) for dim,size in zip(self.dims,self.shape)}
        for dim,factor in factors.items():
            old_edge = old_edges[dim]
            if len(old_edge) - 1 % factor != 0:
                raise ValueError('Input rebinning factor {:d} does not divide data size {:d} along {}'.format(factor,len(old_edge)-1,dim))
            edges[dim] = old_edge[::factor]
        return _matrix_new_edges(old_edges,edges,flatten=flatten,weights=weights)[0]

    def rebin(self, factors, dims=None, weights=None, columns_to_sum=None):
        if weights is None:
            weights = self.attrs.get('weights',None)
        columns_to_sum = columns_to_sum or []

        if weights is not None:
            if isinstance(weights,str):
                columns_to_sum.append(weights)
                weights = self[weights]
        else:
            weights = np.ones(self.shape,dtype='f8')

        matrices = self._matrix_rebin(factors,dims=dims,flatten=False,weights=None)

        binned_weights = _apply_matrices(weights,matrices)
        for col in self.data:
            self.data[col] = _apply_matrices(self.data[col]*weights,matrices)
            if col not in columns_to_sum:
                self.data[col] /= binned_weights

    def squeeze(self, dims=None):
        if dims is None:
            dims = [dim for idim,dim in enumerate(self.dims) if self.shape[idim] <= 1]
        if np.ndim(dims) == 0:
            dims = [dims]
        axes = tuple([self.dims.index(dim) for dim in dims])
        for col in self.data:
            self.data[col] = np.squeeze(self.data[col],axis=axes)
        for dim in dims:
            del self.dims[self.dims.index(dim)]
            if dim in self.edges:
                del self.edges[dim]

    def average(self, dims=None, weights=None, columns_to_sum=None):
        if dims is None:
            dims = self.dims
        if np.ndim(dims) == 0:
            dims = [dims]
        axes = [self.dims.index(dim) for dim in dims]
        factors = {dim:shape[axis] for dim,axis in zip(dims,axes)}
        self.rebin(factors,weights=weights,columns_to_sum=columns_to_sum)
        self.squeeze(dims=dims)

    @classmethod
    def get_title_label(cls):
        return _title_template.format('.'.join((cls.__module__,cls.__name__)))

    @classmethod
    def read_title_label(cls, line):
        template = _title_template.replace('{}','(.*)')
        match = re.match(template,line)
        if match:
            module_class = match.group(1)
            import importlib
            importlib.import_module(re.match('(.*)\.(.*)$',module_class).group(1))
            return cls._registry[module_class]

    @savefile
    def save_txt(self, filename=None, fmt='.18e', comments='#', ignore_json_errors=True):
        lines = self.get_header_txt(comments=comments,ignore_json_errors=ignore_json_errors)
        data = {col:self.data[col].flatten() for col in self.columns}
        for ii in range(self.size):
            lines.append(' '.join(['{:{fmt}}'.format(data[col][ii],fmt=fmt) for col in data]))
        if filename is not None:
            if self.is_mpi_root():
                with open(filename,'w') as file:
                    for line in lines:
                        file.write(line + '\n')
        else:
            return lines

    def get_header_txt(self, comments='#', ignore_json_errors=True):
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
    def read_header_txt(cls, file, comments='#', mapping_header=None, pattern_header=None, ignore_json_errors=True):
        attrs = {}
        mapping_header = (mapping_header or {}).copy()
        mapping_header = utils.dict_nonedefault(mapping_header,**cls._default_mapping_header)

        def decode_value(value, decode):
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

        def fill_attrs_values(key, *values):
            if len(values) > 1:
                name,value = values
                if key not in attrs:
                    attrs[key] = {}
                attrs[key][name] = decode_value(value,decode)
            else:
                attrs[key] = decode_value(values[0],decode)

        for line in file:
            if not line.startswith(comments):
                break
            for key,pattern in mapping_header.items():
                if isinstance(pattern,tuple):
                    pattern,decode = pattern
                else:
                    decode = None
                match = re.match(pattern,line[len(comments):])
                if match is not None:
                    fill_attrs_values(key,*match.groups())
                    cls.log_debug('Setting attribute {} = {} from header.'.format(key,attrs[key]),rank=0)
            if pattern_header:
                match = re.match(pattern_header,line[len(comments):])
                if match is not None:
                    fill_attrs_values(*match.groups())

        for key in mapping_header.items():
            if key not in attrs:
                cls.log_debug('Could not find attribute {} in header'.format(key))

        return attrs

    @classmethod
    def load_txt(cls, filename, comments='#', usecols=None, skip_rows=0, max_rows=None, mapping_header=None, pattern_header=None, attrs=None, **kwargs):

        if isinstance(filename,str):
            cls.log_info('Loading {}.'.format(filename),rank=0)
            file = []
            with open(filename,'r') as file_:
                for line in file_:
                    file.append(line)
        else:
            file = [line for line in filename]

        tmpcls = cls.read_title_label(file[0][len(comments):])
        self_format = tmpcls is not None
        if self_format:
            cls = tmpcls
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
            if iline < skip_rows: continue
            if max_rows is not None and iline >= skip_rows + max_rows: break
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
        new = super(BinnedStatistic,self).__copy__()
        for name in ['data','dims','edges','attrs']:
            setattr(new,name,getattr(new,name).copy())
        return new


class BinnedProjection(BinnedStatistic):
    """
    Structure holding a binned statistic.
    """
    logger = logging.getLogger('BinnedProjection')

    _default_mapping_header = {**BinnedStatistic._default_mapping_header,'proj':'.*?#proj = (.*)$','y':'.*?y = (.*)$'}

    def __init__(self, data=None, x=None, y=None, dims=None, weights=None, proj=None, edges=None, attrs=None):
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
        return all(x in self.data for x in self.dims)

    def has_y(self):
        return 'y' in self.attrs and self.attrs['y'] in self.data

    def get_x_average(self, xlim=None, mask=Ellipsis, weights=None, from_edges=None):
        if from_edges is None: from_edges = not self.has_x()
        masks = self.get_index(xlim=xlim,mask=mask,flatten=False)

        def mid(edges):
            return (edges[:-1] + edges[1:])/2.

        if from_edges:
            return [mid(self.edges[dim])[masks[idim]] for idim,dim in enumerate(self.dims)]

        x = self.get_x(flatten=False)
        allaxes = list(range(self.ndim))
        toret = []
        for idim,dim in enumerate(self.dims):
            axes = allaxes.copy()
            del axes[idim]
            toret.append(np.average(x[idim],axis=axes,weights=weights))
        return toret

    def get_x(self, xlim=None, mask=Ellipsis, flatten=True):
        x = [self[x] for x in self.dims]
        if xlim is None and mask is Ellipsis:
            index = Ellipsis
        else:
            index = self.get_index(xlim,mask=mask,flatten=flatten)
        if flatten:
            x = np.moveaxis([x_.flatten()[index] for x_ in x],0,-1)
            if x.shape[-1] == 1:
                return x[...,0]
            return x
        x = np.moveaxis([x_[index] for x_ in x],0,-1)
        if x.shape[-1] == 1:
            return x[...,0]
        return x

    def get_y(self, xlim=None, mask=Ellipsis, flatten=True):
        y = self[self.attrs['y']]
        if flatten:
            return y.flatten()[self.get_index(xlim,mask=mask,flatten=flatten)]
        return y[self.get_index(xlim,mask=mask,flatten=flatten)]

    def get_edges(self, xlim=None, mask=Ellipsis):
        # mask only in the form (1d, 1d, ...)
        masks = self.get_index(xlim=xlim,mask=mask,flatten=False)
        toret = _mask_edges(self.edges,masks)
        return [toret[dim] for dim in self.dims]

    def set_x(self, x, mask=Ellipsis, flatten=True):
        for ix,x_ in enumerate(self.dims):
            self.data.setdefault(x_,np.full(self.shape,np.nan))
            if flatten:
                self[x_].flat[mask] = x[...,ix] if x.ndim > 1 else x
            else:
                self[x_][mask] = x

    def set_y(self, y, mask=Ellipsis, flatten=True):
        self.attrs.setdefault('y','y')
        self.data.setdefault(self.attrs['y'],np.full(self.shape,np.nan))
        y_ = self.attrs['y']
        if flatten:
            self[y_].flat[mask] = y
        else:
            self[y_][mask] = y

    def get_index(self, xlim=None, mask=Ellipsis, flatten=True):
        if flatten:
            mask_ = np.zeros(self.size,dtype='?')
            mask_[mask] = True
            if xlim is not None:
                x = self.get_x()
                tmp = (x >= xlim[0]) & (x <= xlim[-1])
                if tmp.ndim > 1:
                    tmp = np.all(tmp,axis=-1)
                mask_ &= tmp
            return np.flatnonzero(mask_)
        toret = []
        x = self.get_x(flatten=False)
        allaxes = list(range(self.ndim))
        if mask is not Ellipsis:
            if np.ndim(mask) == 1:
                mask = [mask]*self.ndim
        for idim,dim in enumerate(self.dims):
            mask_ = np.zeros(self.shape[idim],dtype='?')
            mask_[mask[idim]] = True
            if xlim is not None:
                tmp = (x[...,idim] >= xlim[0]) & (x[...,idim] <= xlim[-1])
                axes = allaxes.copy()
                del axes[idim]
                tmp = np.all(tmp,axis=tuple(axes))
                mask_ &= tmp
            toret.append(np.flatnonzero(mask_))
        return toret

    def squeeze(self, dims=None):
        if dims is None:
            dims = [dim for idim,dim in enumerate(self.dims) if self.shape[idim] <= 1]
        if np.ndim(dims) == 0:
            dims = [dims]
        super(BinnedProjection,self).squeeze(dims=dims)
        x = list(self.dims)
        for dim in dims:
            if dim in x:
                del x[x.index(dim)]
        self.dims = x

    def __getstate__(self):
        state = super(BinnedProjection,self).__getstate__()
        state['proj'] = self.proj.__getstate__()
        return state

    def __setstate__(self, state):
        super(BinnedProjection,self).__setstate__(state)
        self.proj = ProjectionName.from_state(self.proj)

    def get_header_txt(self, comments='#', **kwargs):
        header = super(BinnedProjection,self).get_header_txt(comments=comments,**kwargs)
        header += ['{}#proj = {}'.format(comments,json.dumps(self.proj.__getstate__()))]
        return header
