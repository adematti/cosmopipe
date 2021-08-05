"""Definition of :class:`DataVector` to store a data vector."""

import os
import logging
import json

import numpy as np
from scipy import linalg

from cosmopipe.lib.utils import BaseClass, savefile
from cosmopipe.lib import utils
from .binned_statistic import BinnedProjection, get_title_label, read_title_label, read_header_txt
from .projection import ProjectionName, ProjectionNameCollection


_list_types = (ProjectionNameCollection,list,np.ndarray)


def _format_index_kwargs(kwargs):
    # make sure all values of kwargs are list of same size
    toret = {}
    # put lists
    for key,value in kwargs.items():
        if not isinstance(value,_list_types):
            value = [value]
        toret[key] = value
    # check all values are of same length
    if toret:
        n = len(list(toret.values())[0])
        if not all(len(value) == n for value in toret.values()):
            raise IndexError('Input parameters {} have different lengths.'.format(kwargs))
    return toret


def _reduce_index_kwargs(kwargs, projs=None):
    # remove redundant entries in kwargs
    # e.g. take intersection of xlim values
    xlims = {}

    def callback(proj, xlim):
        if proj in xlims:
            if xlims[proj] is None:
                xlims[proj] = xlim
            elif xlim is not None:
                # intersection of xlim
                xlims[proj] = (np.max([xlims[proj][0],xlim[0]]),np.min([xlims[proj][1],xlim[1]]))
        else:
            xlims[proj] = xlim

    kwargs = kwargs.copy()
    if not kwargs: return {'proj':projs,'xlim':[None]*len(projs)}

    # fill in proj and xlim keys
    if 'proj' not in kwargs:
        kwargs['proj'] = [None]*len(kwargs['xlim'])
    if 'xlim' not in kwargs:
        kwargs['xlim'] = [None]*len(kwargs['proj'])

    for proj,xlim in zip(kwargs['proj'],kwargs['xlim']):
        if proj is None: # if proj value is None, requires projs, input list of data projections
            for proj in projs:
                callback(proj,xlim)
        else:
            callback(proj,xlim)

    # reduced set of limits
    toret = {'proj':[],'xlim':[]}
    for proj,xlim in xlims.items():
        toret['proj'].append(proj)
        toret['xlim'].append(xlim)
    return toret


def _getstate_index_kwargs(**kwargs):
    # return view state
    if 'proj' in kwargs:
        kwargs['proj'] = [ProjectionName(proj).__getstate__() if proj is not None else None for proj in kwargs['proj']]
    return kwargs


def _setstate_index_kwargs(**kwargs):
    # set view state
    if 'proj' in kwargs:
        kwargs['proj'] = [ProjectionName.from_state(proj) if proj is not None else None for proj in kwargs['proj']]
    return kwargs


class RegisteredDataVector(type):

    """Metaclass registering :class:`DataVector` derived classes."""

    _registry = {}

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)
        meta._registry['.'.join((class_dict['__module__'],name))] = cls
        return cls


class DataVector(BaseClass,metaclass=RegisteredDataVector):
    """
    Class representing a data vector, as a collection of :class:`BinnedProjection`,
    e.g. power spectrumm multipoles, wedges, etc.
    One can also store heterogenous quantities, e.g. power spectrum, correlation function,
    higher order...

    Attributes
    ----------
    data : dict
        Dictionary of :attr:`BinnedProjection` instances.

    attrs : dict
        Dictionary of other attributes.
    """
    logger = logging.getLogger('DataVector')

    _title_template = '### {} ###'
    _default_mapping_header = {'kwargs_view':'.*?#kwview = (.*)$'}

    def __init__(self, x=None, y=None, proj=None, edges=None, attrs=None, **kwargs):
        """
        Initialize data vector.

        Parameters
        ----------
        x : DataVector, BinnedProjection, list, array
            If list of arrays, x-coordinates for all projections.
            If single array, is repeated for each projection.
            If (list of) :class:`BinnedProjection`: set these projection(s).
            If :class:`DataVector`, is (shallow) copied.

        y : list, default=None
            y-coordinates for all projections.
            If ``None``, no y-coordinates set.

        proj : list, default=None
            List of (inputs for) projection names :class:`ProjectionName`.

        edges : list, default=None
            If ``None``, no edges set.

        attrs : dict
            Dictionary of other attributes.

        kwargs : dict
            Other arguments for :class:`BinnedProjection` (same for all of them).
        """
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

    def get_index(self, permissive=True, index_in_view=False, **kwargs):
        """
        Return indices corresponding to input selections (including current view).

        Example
        -------
        ``pk.get_index(xlim=(0.1,0.2),proj=0)`` will return index to obtain ``pk`` monopole between ``0.1`` and ``0.2``.
        ``pk.get_index(proj=(0,2))`` will return indices to obtain ``pk`` monopole and quadrupole.

        Parameters
        ----------
        permissive : bool
            If ``True``, include projections which match input ``proj`` (in ``kwargs``) for non-``None`` attributes.
            For example `ProjectionName(space='power',mode='multipole',proj=0)` would match ``ell_0`` (``space`` not specified).

        index_in_view : bool
            Whether to return index w.r.t. to current view.

        kwargs : dict
            Dictionary of selections (i.e. list of ``proj`` and ``xlim``).

        Returns
        -------
        indices : tuple, list
            Tuple (if input selections are not lists), or list of tuples ``(proj, index)``
            with ``index`` indices for projection of name ``proj``.
        """
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
            if isinstance(value,_list_types):
                isscalar = False
                break

        kwargs = _format_index_kwargs(kwargs)
        for ii in range(len(list(kwargs.values())[0])):
            index += _get_one_index(**{key:value[ii] for key,value in kwargs.items()},permissive=permissive)
        if isscalar and index:
            index = index[0]

        return index

    def get_x(self, concatenate=False, **kwargs):
        """
        Return x-coordinate of the data vector.

        Example
        -------
        ``pk.get_x(xlim=(0.1,0.2),proj=0)`` will return wavenumbers for ``pk`` monopole between ``0.1`` and ``0.2``.
        ``pk.get_x(proj=(0,2))`` will return wavenumbers for ``pk`` monopole and quadrupole.

        Parameters
        ----------
        concatenate : bool, default=False
            Concatenate output x-coordinates?

        kwargs : dict
            Dictionary of selections (i.e. list of ``proj`` and ``xlim``).

        Warning
        -------
        Output x-coordinates must be of same dimensionality to be concatenated!
        """
        index = self.get_index(**kwargs)
        if not isinstance(index,list):
            return self.get(index[0]).get_x(mask=index[1])
        x = [self.get(proj).get_x(mask=index_) for proj,index_ in index]
        if concatenate:
            return np.concatenate(x)
        return x

    def get_y(self, concatenate=True, **kwargs):
        """
        Return y-coordinate of the data vector.
        Same as :meth:`get_x`, but for the y-coordinate (which is by definition of 1 dimension for all projections).
        """
        index = self.get_index(**kwargs)
        if not isinstance(index,list):
            return self.get(index[0]).get_y(mask=index[1])
        y = [self.get(proj).get_y(mask=index_) for proj,index_ in index]
        if concatenate:
            return np.concatenate(y)
        return y

    def get_edges(self, **kwargs):
        """Return edges for ``kwargs`` selections."""
        index = self.get_index(**kwargs)
        if not isinstance(index,list):
            return self.get(index[0]).get_edges(mask=index[1])
        return [self.get(proj).get_edges(mask=index_) for proj,index_ in index]

    def __len__(self):
        """Return data vector total length in this view."""
        index = self.get_index()
        return sum(index_.size for proj,index_ in index)

    @property
    def size(self):
        """Equivalent for :meth:len."""
        return len(self)

    @property
    def projs(self):
        """Return projection names in data vector."""
        return ProjectionNameCollection([dataproj.proj for dataproj in self.data])

    def get_projs(self, **kwargs):
        """Return projection names in data vector, after selections."""
        index = self.get_index(**kwargs)
        return ProjectionNameCollection([proj for proj,index_ in index if index_.size])

    def view(self, **kwargs):
        """
        Set data vector view, i.e. ensemble of selections to apply.
        This will apply to output of ``get_`` methods.

        Example
        -------
        ``pk.view(xlim=(0.1,0.2)).get_y(proj=0)`` will return ``pk`` monopole between ``0.1`` and ``0.2``.
        """
        self._kwargs_view = _reduce_index_kwargs(_format_index_kwargs(kwargs),projs=self.projs)
        #self._kwargs_view = _format_index_kwargs(kwargs)
        return self

    def noview(self):
        """Reset view."""
        self._kwargs_view = None
        return self

    @property
    def kwview(self):
        """Current view selections."""
        return self._kwargs_view or {}

    def __contains__(self, proj):
        """Whether data vector contains projection of name ``proj``."""
        return proj in self.projs

    def set_new_edges(self, proj, *args, **kwargs):
        """
        Set new edges for projection of name ``proj`` (list or single projection name).
        See :meth:`BinnedStatistic.set_new_edges` for other arguments.
        """
        if not isinstance(proj,list):
            proj = [proj]
        for proj_ in proj:
            self.get(proj_).set_new_edges(*args,**kwargs)

    def _matrix_new_edges(self, proj, *args, flatten=False, **kwargs):
        # return matrix transform for whole data vector
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
        """
        Rebin projection of name ``proj`` (list or single projection name).
        See :meth:`BinnedStatistic.rebin` for other arguments.
        """
        if not isinstance(proj,list):
            proj = [proj]
        for proj_ in proj:
            self.get(proj_).rebin(*args,**kwargs)

    def _matrix_rebin(self, proj, *args, flatten=False, **kwargs):
        # return matrix transform for whole data vector
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
        """Add new :class:`BinnedProjection` instance."""
        if not isinstance(item,BinnedProjection):
            raise TypeError('{} is not a BinnedProjection instance.'.format(item))
        proj = ProjectionName(name)
        if proj != item.proj:
            raise KeyError('BinnedProjection {} should be indexed by proj (incorrect {})'.format(item,proj))
        self.data[self.projs.index(proj)] = item

    def get(self, proj, permissive=False):
        """
        Return :class:`BinnedProjection` instance corresponding to ``proj``.

        Parameters
        ----------
        proj : ProjectionName, tuple, string, dict
            Projection name to search for.

        permissive : bool, default=False
            If ``True``, include projections which match input ``proj`` (in ``kwargs``) for non-``None`` attributes.
            For example `ProjectionName(space='power',mode='multipole',proj=0)` would match ``ell_0`` (``space`` not specified).

        Returns
        -------
        dataproj : BinnedProjection
        """
        if permissive:
            return [self.data[index] for index in self.projs.index(proj,ignore_none=permissive)]
        return self.data[self.projs.index(proj)]

    def set(self, data):
        """Add new :class:`BinnedProjection` instance."""
        if not isinstance(data,BinnedProjection):
            raise TypeError('{} is not a BinnedProjection instance.'.format(data))
        if data.proj in self:
            self[data.proj] = data
        else:
            self.data.append(data)

    def set_y(self, y, concatenated=True, **kwargs):
        """
        Set y-coordinates.

        Example
        -------
        ``pk.view(xlim=(0.1,0.2)).set_y(y)`` will set y between ``0.1`` and ``0.2``.

        Parameters
        ----------
        y : list, array
            y arrays for each projection.

        concatenated : bool, default=True
            If ``True``, ``y`` is a single array, which is split
            into the different projections given current view and ``kwargs``.

        kwargs : dict
            Arguments for :meth:`get_index`
        """
        start = 0
        for iproj,(proj,index) in enumerate(self.get_index(**kwargs)):
            dataproj = self.get(proj)
            if concatenated:
                stop = start + index.size
                dataproj.set_y(y[start:stop],mask=index)
                start = stop
            else:
                dataproj.set_y(y[iproj],mask=index)

    def copy(self, copy_proj=False):
        """Return copy, including shallow-copy of each projection."""
        new = self.__copy__()
        if copy_proj:
            new.data = [dataproj.copy() for dataproj in self.data]
        return new

    def __copy__(self):
        """Return copy (without copying each projection)."""
        new = super(DataVector,self).__copy__()
        for name in ['data','attrs']:
            setattr(new,name,getattr(new,name).copy())
        if self._kwargs_view is not None:
            new._kwargs_view = self._kwargs_view.copy()
        return new

    def __getstate__(self):
        """Return this class state dictionary."""
        data = []
        for projection in self.data:
            data.append(projection.__getstate__())
        state = {'data':data,'attrs':self.attrs}
        state['_kwargs_view'] = _getstate_index_kwargs(**self._kwargs_view) if self._kwargs_view is not None else None
        state['__class__'] = '.'.join((self.__class__.__module__,self.__class__.__name__))
        return state

    def __setstate__(self, state):
        """Set this class state dictionary."""
        if isinstance(state['data'],list):
            self.data = [BinnedProjection.from_state(binned_projection) for binned_projection in state['data']]
            self.attrs = state['attrs']
            self._kwargs_view = _setstate_index_kwargs(**state['_kwargs_view']) if state['_kwargs_view'] is not None else None
        else:
            binned_projection = BinnedProjection.from_state(state['data'])
            self.set(binned_projection)
            self._kwargs_view = None

    @classmethod
    def concatenate(cls, *others):
        """
        Concatenate data vectors together.

        Parameters
        ----------
        others : list
            List of :class:`DataVector` instances.

        Returns
        -------
        new : DataVector

        Warning
        -------
        :attr:`attrs` of returned data vector contains, for each key, the last value found in ``others`` :attr:`attrs` dictionaries.
        """
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
        """Extend data vector with ``other``."""
        new = self.concatenate(self,other)
        self.__dict__.update(new.__dict__)

    def __radd__(self, other):
        """Operation corresponding to ``other + self``"""
        if other in [[],0,None]:
            return self.copy()
        return self.concatenate(self,other)

    def __add__(self, other):
        """Addition of two data vectors is defined as concatenation."""
        return self.concatenate(self,other)

    def plot(self, style=None, **kwargs_style):
        """Plot data vector. See :class:`plotting.DataPlotStyle`."""
        from .plotting import DataPlotStyle
        style = DataPlotStyle(style=style,data_vectors=self,**kwargs_style)
        style.plot()

    @classmethod
    def load_auto(cls, filename, *args, **kwargs):
        """
        Load data vector.

        Note
        ----
        Returned data vector, if saved with a :class:`DataVector`-inherited class, will be an instance of that class.

        Parameters
        ----------
        filename : string
            File name of data vector.
            If ends with '.txt', call :meth:`load_txt`
            Else (numpy binary format), call :meth:`load`

        args : list
            Arguments for load function.

        kwargs : dict
            Other arguments for load function.
        """
        if os.path.splitext(filename)[-1] == '.txt':
            return cls.load_txt(filename,*args,**kwargs)
        return cls.load(filename)

    def save_auto(self, filename, *args, **kwargs):
        """
        Write data vector to disk.

        Parameters
        ----------
        filename : string
            File name of data vector.
            If ends with '.txt', call :meth:`save_txt`
            Else (numpy binary format), call :meth:`save`

        args : list
            Arguments for save function.

        kwargs : dict
            Other arguments for save function.
        """
        if os.path.splitext(filename)[-1] == '.txt':
            return self.save_txt(filename,*args,**kwargs)
        return self.save(filename)

    def get_header_txt(self, comments='#', ignore_json_errors=True):
        """
        Dump header:

        - items in :attr:`attrs`
        - :attr:`_kwargs_view` (current view selections)

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
        if self._kwargs_view is not None:
            header.append('{}{} = {}'.format(comments,'#kwview',json.dumps(_getstate_index_kwargs(**self._kwargs_view))))
        return header

    @classmethod
    def load_txt(cls, filename, comments='#', usecols=None, skip_rows=0, max_rows=None, mapping_header=None, columns=None, mapping_proj=None, attrs=None, **kwargs):
        """
        Load :class:`BinnedStatistic` from disk.

        Note
        ----
        If previously saved using :meth:`save_txt`, loading the :class:`DataVector` only requires ``filename``.
        In this case, the returned instance will be of the class that was used to create it (e.g. :class:`MockDataVector`)
        - not necessarily :class:`DataVector`.

        Parameters
        ----------
        filename : string
            File name to read in.

        comments : string, default='#'
            Characters used to indicate the start of a comment.

        usecols : list, default=None
            Which columns to read, with 0 being the first. If ``None``, reads all columns.

        skip_rows : int, default=0
            Skip the first ``skip_rows`` lines, including comments.

        max_rows : int, default=None
            Read ``max_rows lines`` of content after ``skip_rows`` lines. The default is to read all the lines.

        mapping_header : dict, default=None
            Dictionary holding key:regex mapping or (regex, type) to provide the type.
            The corresponding values, read in the header, will be saved in the :attr:`attrs` dictionary.

        columns : list, default=None
            column names corresponding to ``usecols``. Columns 'x' and 'y' are used as x- and y-coordinates.

        mapping_proj : dict, list, default=None
            Dictionary holding a mapping from column name to projection specifier (e.g. 'ell_0', ['muwedge', [0.0,0.2]], or with a name, e.g.: 'ELG_ell_0', ['ELG','muwedge',[0.0,0.2]]).
            It can also be a list corresponding to input columns (skipping the first - x)."

        attrs : dict, default=None
            Attributes to save in the :attr:`attrs` dictionary.

        kwargs : dict
            Other arguments for :meth:`BinnedProjection.load_txt`.

        Returns
        -------
        data : DataVector
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

        tmpcls = cls.read_title_label(file[0][len(comments):])
        self_format = tmpcls is not None
        if self_format:
            cls = tmpcls
            iline_seps = [0]
            for iline,line in enumerate(file[1:]):
                if line.startswith(comments):
                    if BinnedProjection.read_title_label(line[len(comments):]):
                        iline_seps.append(iline+1)
            iline_seps.append(len(file))
        else:
            cls.log_info('Not in {} standard format.'.format(cls.__class__.__name__),rank=0)
            iline_seps = [0,len(file)]
        attrs = (attrs or {}).copy()
        kwargs_view = None

        header = cls.read_header_txt(file[iline_seps[0]:iline_seps[1]],comments=comments,mapping_header=mapping_header)
        kwargs_view = header.pop('kwargs_view',None)
        attrs = utils.dict_nonedefault(attrs,**header)
        if self_format:
            iline_seps = iline_seps[1:]

        new = cls.__new__(cls)
        DataVector.__init__(new,attrs=attrs)
        new._kwargs_view = _setstate_index_kwargs(**kwargs_view) if kwargs_view is not None else None
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
                if data.dims == ['x']: # default value
                    data.dims = [columns[0]]
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
        """
        Dump :class:`DataVector`.

        Parameters
        ----------
        filename : string, default=None
            ASCII file name where to save data vector.
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
        for proj in self.projs:
            lines += self.get(proj).save_txt(filename=None,comments=comments,fmt=fmt,ignore_json_errors=ignore_json_errors)
        if filename is not None:
            if self.is_mpi_root():
                with open(filename,'w') as file:
                    for line in lines:
                        file.write(line + '\n')
        return lines


DataVector.get_title_label = get_title_label
DataVector.read_title_label = read_title_label
DataVector.read_header_txt = read_header_txt
