import os
import logging
import json

import numpy as np
from scipy import linalg

from cosmopipe.lib.utils import BaseClass, savefile
from cosmopipe.lib import utils
from .binned_statistic import BinnedProjection, _title_template
from .projection import ProjectionName


def _format_index_kwargs(kwargs):
    toret = {}
    #kwargs = kwargs or {'proj':None,'xlim':None}
    for key,value in kwargs.items():
        if not isinstance(value,(list,np.ndarray)):
            value = [value]
        toret[key] = value
    if toret:
        n = len(list(toret.values())[0])
        if not all(len(value) == n for value in toret.values()):
            raise IndexError('Input parameters {} have different lengths.'.format(kwargs))
    return toret


def _reduce_index_kwargs(kwargs, projs=None):
    xlims = {}

    def callback(proj, xlim):
        if proj in xlims:
            if xlims[proj] is None:
                xlims[proj] = xlim
            elif xlim is not None:
                xlims[proj] = (np.max([xlims[proj][0],xlim[0]]),np.min([xlims[proj][1],xlim[1]]))
        else:
            xlims[proj] = xlim

    kwargs = kwargs.copy()
    if not kwargs: return {'proj':projs,'xlim':[None]*len(projs)}

    if 'proj' not in kwargs:
        kwargs['proj'] = [None]*len(kwargs['xlim'])
    if 'xlim' not in kwargs:
        kwargs['xlim'] = [None]*len(kwargs['proj'])

    for proj,xlim in zip(kwargs['proj'],kwargs['xlim']):
        if proj is None:
            for proj in projs:
                callback(proj,xlim)
        else:
            callback(proj,xlim)

    toret = {'proj':[],'xlim':[]}
    for proj,xlim in xlims.items():
        toret['proj'].append(proj)
        toret['xlim'].append(xlim)
    return toret


def _getstate_index_kwargs(**kwargs):
    if 'proj' in kwargs:
        kwargs['proj'] = [ProjectionName(proj).__getstate__() if proj is not None else None for proj in kwargs['proj']]
    return kwargs


def _setstate_index_kwargs(**kwargs):
    if 'proj' in kwargs:
        kwargs['proj'] = [ProjectionName.from_state(proj) if proj is not None else None for proj in kwargs['proj']]
    return kwargs


class DataVector(BaseClass):

    logger = logging.getLogger('DataVector')

    _default_mapping_header = {'kwargs_view':'.*?kwview = (.*)$'}

    def __init__(self, x=None, y=None, proj=None, edges=None, attrs=None, **kwargs):
        if isinstance(x,self.__class__):
            self.__dict__.update(x.__dict__)
            return
        self.data = []
        self.attrs = attrs or {}
        self.noview()
        if x is None and y is None:
            return
        if isinstance(x,BinnedProjection):
            self.set(x)
            return
        if x is not None and isinstance(x[0],BinnedProjection):
            for x_ in x: self.set(x_)
            return
        n = 1
        for value in [y,proj,edges]:
            if value is not None:
                n = len(value)
                break
        if x is not None and np.ndim(x[0]) == 0:
            x = [x]*n
        if x is None: x = [x]*n
        if y is None: y = [y]*n
        if proj is None: proj = [proj]*n
        if edges is None: edges = [edges]*n
        #if y is not None and np.ndim(y[0]) != 0:
        #    if x is None or np.ndim(x[0]) == 0:
        #        x = [x]*len(y)
        #    if edges is None or np.ndim(edges[0]) == 0:
        #        edges = [edges]*len(y)
        #else:
        #    x,y,proj,edges = [x],[y],[proj],[edges]
        for x_,y_,proj_,edges_ in zip(x,y,proj,edges):
            self.set(BinnedProjection(x=x_,y=y_,proj=proj_,edges=edges_,**kwargs))

    @classmethod
    def get_title_label(cls):
        return '### {} ###'.format(cls.__name__)

    def get_index(self, permissive=True, index_in_view=False, **kwargs):

        index_view = None

        if self._kwargs_view is not None:
            kwargs_view = self._kwargs_view
            self._kwargs_view = None
            index_view = self.get_index(**kwargs_view,permissive=permissive)
            self._kwargs_view = kwargs_view
            index_view_proj = {}
            for proj,index_ in index_view:
                if proj not in index_view_proj:
                    index_view_proj[proj] = []
                index_view_proj[proj].append(index_)
            for proj,index_ in index_view_proj.items():
                index_view_proj[proj] = np.concatenate(index_)

        def _get_one_index(xlim=None, proj=None, permissive=False):
            proj = ProjectionName(proj)
            if permissive:
                return sum([_get_one_index(xlim=xlim,proj=dataproj.proj,permissive=False) for dataproj in self.get(proj,permissive=permissive)],[])
            dataproj = self.get(proj)
            proj = dataproj.proj
            index = dataproj.get_index(xlim=xlim)
            if index_view is not None:
                if proj in index_view_proj:
                    index = index[np.isin(index,index_view_proj[proj])]
                    if index_in_view:
                        argsort = np.argsort(index_view_proj[proj])
                        index_view_sorted = index_view_proj[proj][argsort]
                        index = argsort[np.searchsorted(index_view_sorted,index)]
                else:
                    return []
            return [(proj,index)]

        if not kwargs:
            #if index_view is not None:
            #    return index_view
            return _get_one_index(permissive=permissive)

        index = []
        isscalar = kwargs.get('proj',None) is not None
        for key,value in kwargs.items():
            if isinstance(value,(list,np.ndarray)):
                isscalar = False
                break

        kwargs = _format_index_kwargs(kwargs)
        for ii in range(len(list(kwargs.values())[0])):
            index += _get_one_index(**{key:value[ii] for key,value in kwargs.items()},permissive=permissive)
        if isscalar and index:
            index = index[0]
        return index

    def get_x(self, concatenate=False, **kwargs):
        """Return x-coordinate of the data vector."""
        index = self.get_index(**kwargs)
        if not isinstance(index,list):
            return self.get(index[0]).get_x(mask=index[1])
        x = [self.get(proj).get_x(mask=index_) for proj,index_ in index]
        if concatenate:
            return np.concatenate(x)
        return x

    def get_y(self, concatenate=True, **kwargs):
        """Return y-coordinate of the data vector."""
        index = self.get_index(**kwargs)
        if not isinstance(index,list):
            return self.get(index[0]).get_y(mask=index[1])
        y = [self.get(proj).get_y(mask=index_) for proj,index_ in index]
        if concatenate:
            return np.concatenate(y)
        return y

    def get_edges(self, **kwargs):
        index = self.get_index(**kwargs)
        if not isinstance(index,list):
            return self.get(index[0]).get_edges(mask=index[1])
        return [self.get(proj).get_edges(mask=index_) for proj,index_ in index]

    def __len__(self):
        index = self.get_index()
        return sum(index_.size for proj,index_ in index)

    @property
    def size(self):
        return len(self)

    @property
    def projs(self):
        return [dataproj.proj for dataproj in self.data]

    def get_projs(self, **kwargs):
        index = self.get_index(**kwargs)
        return [proj for proj,index_ in index if index_.size]

    def view(self, **kwargs):
        self._kwargs_view = _reduce_index_kwargs(_format_index_kwargs(kwargs),projs=self.projs)
        #self._kwargs_view = _format_index_kwargs(kwargs)
        return self

    def noview(self):
        self._kwargs_view = None
        return self

    @property
    def kwview(self):
        return self._kwargs_view or {}

    def __contains__(self, projection):
        return ProjectionName(projection) in self.projs

    def get_proj_index(self, proj, permissive=False):
        proj = ProjectionName(proj)
        self_projs = self.projs
        if permissive:
            return [iproj_ for iproj_,proj_ in enumerate(self_projs) if proj.eq_ignore_none(proj_)]
        if proj not in self_projs:
            raise KeyError('Projection {} not found among {}'.format(proj,self_projs))
        return self_projs.index(proj)

    def set_new_edges(self, proj, *args, **kwargs):
        if not isinstance(proj,list):
            proj = [proj]
        for proj_ in proj:
            self.get(proj_).set_new_edges(*args,**kwargs)

    def _matrix_new_edges(self, proj, *args, flatten=False, **kwargs):
        if not isinstance(proj,list):
            proj = [proj]
        toret = {proj_:np.eye(self.get(proj_).size) for proj_ in self.projs}
        for proj_ in proj:
            dataproj = self.get(proj_)
            proj_ = dataproj.proj
            toret[proj_] = dataproj._matrix_new_edges(*args,flatten=True,**kwargs)
        index = self.get_index()
        for proj_,index_ in index:
            toret[proj_] = toret[proj_][np.ix_(index_,index_)]
        if flatten:
            toret = linalg.block_diag(*[toret[proj_] for proj_ in self.projs])
        return toret

    def rebin(self, proj, *args, **kwargs):
        if not isinstance(proj,list):
            proj = [proj]
        for proj_ in proj:
            self.get(proj_).rebin(*args,**kwargs)

    def _matrix_rebin(self, proj, *args, flatten=False, **kwargs):
        if not isinstance(proj,list):
            proj = [proj]
        toret = {proj_:np.eye(self.get(proj_).size) for proj_ in self.projs}
        for proj_ in proj:
            dataproj = self.get(proj_)
            proj_ = dataproj.proj
            toret[proj_] = dataproj._matrix_rebin(*args,flatten=True,**kwargs)
        index = self.get_index()
        for proj_,index_ in index:
            toret[proj_] = toret[proj_][np.ix_(index_,index_)]
        if flatten:
            toret = linalg.block_diag(*[toret[proj_] for proj_ in self.projs])
        return toret

    def __setitem__(self, name, item):
        if not isinstance(item,BinnedProjection):
            raise TypeError('{} is not a BinnedProjection instance.'.format(item))
        proj = ProjectionName(name)
        if proj != item.proj:
            raise KeyError('BinnedProjection {} should be indexed by proj (incorrect {})'.format(item,proj))
        self.data[self.get_proj_index(proj)] = item

    def get(self, proj, permissive=False):
        if permissive:
            return [self.data[index] for index in self.get_proj_index(proj,permissive=permissive)]
        return self.data[self.get_proj_index(proj)]

    def set(self, data):
        if not isinstance(data,BinnedProjection):
            raise TypeError('{} is not a BinnedProjection instance.'.format(data))
        if data.proj in self:
            self[data.proj] = data
        else:
            self.data.append(data)

    def set_y(self, y, concatenated=True):
        start = 0
        for iproj,(proj,index) in enumerate(self.get_index()):
            dataproj = self.get(proj)
            if concatenated:
                stop = start + index.size
                dataproj.set_y(y[start:stop],mask=index)
                start = stop
            else:
                dataproj.set_y(y[iproj],mask=index)

    def copy(self, copy_proj=False):
        new = self.__copy__()
        if copy_proj:
            new.data = [dataproj.copy() for dataproj in self.data]
        return new

    def __copy__(self):
        new = super(DataVector,self).__copy__()
        for name in ['data','attrs']:
            setattr(new,name,getattr(new,name).copy())
        if self._kwargs_view is not None:
            new._kwargs_view = self._kwargs_view.copy()
        return new

    def __getstate__(self):
        data = []
        for projection in self.data:
            data.append(projection.__getstate__())
        state = {'__class__':self.__class__.__name__,'data':data,'attrs':self.attrs}
        state['_kwargs_view'] = _getstate_index_kwargs(**self._kwargs_view) if self._kwargs_view is not None else None
        return state

    def __setstate__(self, state):
        if state['__class__'] != self.__class__.__name__:
            binned_projection = BinnedProjection.from_state(state['__class__'])
            self.set(binned_projection)
            self._kwargs_view = None
        else:
            self.data = [BinnedProjection.from_state(binned_projection) for binned_projection in state['data']]
            self.attrs = state['attrs']
            self._kwargs_view = _setstate_index_kwargs(**state['_kwargs_view']) if state['_kwargs_view'] is not None else None

    @classmethod
    def concatenate(cls, *others):
        """WARNING: output attrs merges each attrs."""

        def default_kwview(data):
            return {'proj':[proj for proj in data.projs],'xlim':[None]*len(data.projs)}

        def merge_kwview(kw1, kw2):
            toret = {'proj':[],'xlim':[]}
            for kw in [kw1,kw2]:
                for proj,xlim in zip(kw['proj'],kw['xlim']):
                    toret['proj'].append(proj)
                    toret['xlim'].append(xlim)
            return toret

        new = cls(others[0])
        new.attrs = others[0].attrs.copy()
        has_view = others[0]._kwargs_view is not None
        kwargs_view = others[0]._kwargs_view or default_kwview(others[0])
        for other in others[1:]:
            for proj in kwargs_view['proj']:
                new.get(proj,permissive=True)[0].proj
            projs_view = [new.get(proj,permissive=True)[0].proj for proj in kwargs_view['proj']]
            kwargs_view_ = other._kwargs_view or default_kwview(other)
            projs_view_ = [other.get(proj,permissive=True)[0].proj for proj in kwargs_view_['proj']]
            set_view_ = []
            for proj in other.projs:
                if proj in new.projs:
                    if proj in projs_view_:
                        xlim_ = kwargs_view_['xlim'][projs_view_.index(proj)]
                    else:
                        xlim_ = (1.,-1.)
                    if proj in projs_view:
                        xlim = kwargs_view['xlim'][projs_view.index(proj)]
                        if xlim is None and xlim_ is not None:
                            xlim = xlim_
                        if xlim is not None and xlim_ is not None:
                            xlim = (np.max([xlim_[0],xlim[0]]),np.min([xlim_[1],xlim[1]]))
                        kwargs_view['xlim'][projs_view.index(proj)] = xlim
                else:
                    new.set(other.get(proj))
                    set_view_.append(proj)

            for iproj,proj in enumerate(projs_view_):
                if proj in set_view_:
                    kwargs_view['proj'].append(kwargs_view_['proj'][iproj])
                    kwargs_view['xlim'].append(kwargs_view_['xlim'][iproj])
            has_view = has_view or other._kwargs_view is not None
            new.attrs.update(other.attrs)
        if has_view:
            new._kwargs_view = kwargs_view
        return new

    def extend(self, other):
        new = self.concatenate(self,other)
        self.__dict__.update(new.__dict__)

    def __radd__(self, other):
        if other in [[],0,None]:
            return self.copy()
        return self.concatenate(self,other)

    def __add__(self, other):
        return self.concatenate(self,other)

    def plot(self, style=None, **kwargs_style):
        from .plotting import DataPlotStyle
        style = DataPlotStyle(style=style,data_vectors=self,**kwargs_style)
        style.plot()

    @classmethod
    def load_auto(cls, filename, *args, **kwargs):
        if os.path.splitext(filename)[-1] == '.txt':
            return cls.load_txt(filename,*args,**kwargs)
        return cls.load(filename)

    def save_auto(self, filename, *args, **kwargs):
        if os.path.splitext(filename)[-1] == '.txt':
            return self.save_txt(filename,*args,**kwargs)
        return self.save(filename)

    @classmethod
    def get_title_label(cls):
        return _title_template.format(cls.__name__)

    def get_header_txt(self, comments='#', ignore_json_errors=True):
        header = ['{}{}'.format(comments,self.get_title_label())]
        for key,value in self.attrs.items():
            try:
                header.append('{}{} = {}'.format(comments,key,json.dumps(value)))
            except TypeError:
                if not ignore_json_errors:
                    raise
        if self._kwargs_view is not None:
            header.append('{}{} = {}'.format(comments,'kwview',json.dumps(_getstate_index_kwargs(**self._kwargs_view))))
        return header

    @classmethod
    def load_txt(cls, filename, comments='#', usecols=None, skip_rows=0, max_rows=None, mapping_header=None, mapping_proj=None, columns=None, attrs=None, **kwargs):
        if isinstance(filename,str):
            cls.log_info('Loading {}.'.format(filename),rank=0)
            file = []
            with open(filename,'r') as file_:
                for line in file_:
                    file.append(line)
        else:
            file = [line for line in filename]

        self_format = file[skip_rows].strip() == '{}{}'.format(comments,cls.get_title_label())
        if max_rows is None: max_rows = len(file)
        if self_format:
            iline_seps = [skip_rows]
            for iline,line in enumerate(file):
                if iline > max_rows:
                    break
                if iline > skip_rows and line.startswith(comments):
                    if BinnedProjection.read_title_label(line[len(comments):]):
                        iline_seps.append(iline)
            iline_seps.append(iline+1)
        else:
            iline_seps = [skip_rows,max_rows]
        attrs = (attrs or {}).copy()
        kwargs_view = None

        header = cls.read_header_txt(file[iline_seps[0]:iline_seps[1]],comments=comments,mapping_header=mapping_header)
        kwargs_view = header.pop('kwargs_view',None)
        attrs = utils.dict_nonedefault(attrs,**header)
        if self_format:
            iline_seps = iline_seps[1:]

        new = cls(attrs=attrs)
        new._kwargs_view = kwargs_view
        for start,stop in zip(iline_seps[:-1],iline_seps[1:]):
            projection_format = BinnedProjection.read_title_label(file[start][len(comments):])
            if not projection_format:
                if usecols is None:
                    for line in file[start:stop]:
                        if line.startswith(comments): continue
                        usecols = range(len(line.strip().split()))
                        break
                    if usecols is None:
                        raise ValueError('No columns found in file')
                if columns is None: columns = ['x'] + ['col{:d}'.format(icol) for icol in usecols[1:]]
            data = BinnedProjection.load_txt(file[start:stop],comments=comments,usecols=usecols,mapping_header=mapping_header,columns=columns,**kwargs)
            if not projection_format:
                data.attrs.setdefault('x',(columns[0],))
                data.attrs.setdefault('y',columns[1])
            if mapping_proj is not None:
                #columns = list(mapping_proj.keys())
                if isinstance(mapping_proj,list):
                    if columns is None:
                        raise ValueError('List of columns should be provided if mapping_proj is not None')
                    if len(mapping_proj) != len(columns) - 1:
                        raise ValueError('Provided projection mapping is not of the same length as the number of columns - 1 (for x column). Provide mapping column:projection instead')
                    mapping_proj = {col:mproj for col,mproj in zip(columns[1:],mapping_proj)}
                for col,proj in mapping_proj.items():
                    dataproj = data.copy()
                    dataproj.attrs = {**data.attrs,'y':col}
                    dataproj.proj = ProjectionName(proj)
                    new.set(dataproj)
            else:
                new.set(data)
        return new

    @savefile
    def save_txt(self, filename, comments='#', fmt='.18e', ignore_json_errors=True):
        lines = self.get_header_txt(comments=comments,ignore_json_errors=ignore_json_errors)
        for proj in self.projs:
            lines += self.get(proj).save_txt(filename=None,comments=comments,fmt=fmt,ignore_json_errors=ignore_json_errors)
        if filename is not None:
            if self.is_mpi_root():
                with open(filename,'w') as file:
                    for line in lines:
                        file.write(line + '\n')
        else:
            return lines


DataVector.read_header_txt = BinnedProjection.read_header_txt
