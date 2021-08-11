"""Definition of :class:`CovarianceMatrix` and :class:`MockCovarianceMatrix` to store covariance matrices."""

import os
import logging
import json

import numpy as np

from cosmopipe.lib.utils import BaseClass, savefile
from cosmopipe.lib import utils
from .data_vector import DataVector
from .data_vector import _format_index_kwargs as _format_single_index_kwargs
from .data_vector import _reduce_index_kwargs as _reduce_single_index_kwargs
from .data_vector import _getstate_index_kwargs as _getstate_single_index_kwargs
from .data_vector import _setstate_index_kwargs as _setstate_single_index_kwargs
from .binned_statistic import BinnedProjection, get_title_label, read_title_label, read_header_txt


NDIM = 2


def _getstate_index_kwargs(kwargs):
    # return view state
    return [_getstate_single_index_kwargs(**kw) for kw in kwargs]


def _setstate_index_kwargs(kwargs):
    # set view state
    return [_setstate_single_index_kwargs(**kw) for kw in kwargs]


def _format_index_kwargs(first=None, second=None,  **kwargs):
    # return view for first and second coordinates
    first = first or kwargs
    second = second or first
    return [first,second]


class RegisteredCovarianceMatrix(type):

    """Metaclass registering :class:`CovarianceMatrix` derived classes."""

    _registry = {}

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)
        meta._registry['.'.join((class_dict['__module__'],name))] = cls
        return cls


class CovarianceMatrix(BaseClass,metaclass=RegisteredCovarianceMatrix):
    """
    Class representing a covariance matrix, as an array and two (most often identical)
    data vectors, holding x-coordinates and possibly mean y-coordinates.
    """
    logger = logging.getLogger('CovarianceMatrix')

    _title_template = '### {} ###'
    _default_mapping_header = {'kwargs_view':'.*?#kwview = (.*)$'}

    def __init__(self, covariance, first, second=None, attrs=None):
        """
        Initialize covariance matrix.

        Parameters
        ----------
        covariance : array
            2D array holding covariance.

        first : DataVector
            Data vector for first coordinate in ``covariance``.

        second : DataVector
            Data vector for second coordinate in ``covariance``.

        attrs : dict
            Dictionary of other attributes.
        """
        if isinstance(covariance,self.__class__):
            self.__dict__.update(covariance.__dict__)
            return

        self.attrs = attrs or {}
        self._cov = covariance
        # copy to avoid issues related to changes in _x but not in _cov
        self._x = [DataVector(first).copy()]*NDIM
        if second is not None: self._x[1] = DataVector(second).copy()
        self._kwargs_view = None

    def get_index(self, *args, permissive=True, concatenate=True, **kwargs):
        """
        Return indices corresponding to input selections (including current view).

        Example
        -------
        ``cov.get_index(xlim=(0.1,0.2),proj=0)`` will return index to obtain covariance of monopole between ``0.1`` and ``0.2``.
        ``cov.get_index(proj=(0,2))`` will return indices to obtain covariance of monopole and quadrupole.

        Parameters
        ----------
        permissive : bool, default=True
            If ``True``, include projections which match input ``proj`` (in ``kwargs``) for non-``None`` attributes.
            For example `ProjectionName(space='power',mode='multipole',proj=0)` would match ``ell_0`` (``space`` not specified).

        concatenate : bool, default=True
            If ``True``, return index in full :attr:`cov`.
            Else, return list of ``(proj, index, start)`` tuples, where ``proj`` is projection name,
            index is the index array (with zero being the first index when no x-selection),
            and start the first index of the projection in full :attr:`cov`.

        args : list
            Dictionary(ies) of selections (i.e. list of ``proj`` and ``xlim``) for first and second dimensions.

        kwargs : dict
            Dictionary of selections (i.e. list of ``proj`` and ``xlim``).

        Returns
        -------
        indices : list
            Indices in full :attr:`cov` for first and second dimensions, or list of ``(proj, index, start)`` tuples.
        """
        kwargs = _format_index_kwargs(*args,**kwargs)
        index_view_proj = None
        if self._kwargs_view is not None:
            index_view_proj = [None for axis in range(NDIM)]
            for axis in range(NDIM):
                index_x = self._x[axis].get_index(permissive=permissive,index_in_view=True,**self._kwargs_view[axis])
                index_proj = {}
                for proj,index_ in index_x:
                    if proj not in index_proj:
                        index_proj[proj] = []
                    index_proj[proj].append(index_)
                for proj,index_ in index_proj.items():
                    index_proj[proj] = np.concatenate(index_)
                index_view_proj[axis] = index_proj

        index = [None for axis in range(NDIM)]
        for axis in range(NDIM):
            noview = self._x[axis].get_index(index_in_view=True)
            start, starts = 0, {}
            for proj,index_ in noview:
                starts[proj] = start
                start += index_.size
            index_x = self._x[axis].get_index(permissive=permissive,index_in_view=True,**kwargs[axis])
            isscalar = not isinstance(index_x,list)
            if isscalar: index_x = [index_x]
            index_inview = []
            for proj,index_ in index_x:
                if index_view_proj is not None:
                    if proj in index_view_proj[axis]:
                        index_ = index_[np.isin(index_,index_view_proj[axis][proj])]
                        index_inview.append((proj,index_,starts[proj]))
                else:
                    index_inview.append((proj,index_,starts[proj]))
            if concatenate:
                index_inview = np.concatenate([index[1] + index[2] for index in index_inview])
            elif isscalar:
                index_inview = index_inview[0]
            index[axis] = index_inview
        return index

    def __copy__(self):
        """Return copy (without copying :attr:`cov`)."""
        new = super(CovarianceMatrix,self).__copy__()
        for name in ['attrs']:
            setattr(new,name,getattr(new,name).copy())
        new._x = [x.copy() for x in self._x]
        if self._kwargs_view is not None:
            new._kwargs_view = self._kwargs_view.copy()
        return new

    def view(self, *args, **kwargs):
        """
        Set covariance matrix view, i.e. ensemble of selections to apply.
        This will apply to output of ``get_`` methods.

        Example
        -------
        ``cov.view(xlim=(0.1,0.2)).get_cov(proj=0)`` will return monopole covariance between ``0.1`` and ``0.2``.
        """
        list_kwargs = _format_index_kwargs(*args, **kwargs)
        self._kwargs_view = [_reduce_single_index_kwargs(_format_single_index_kwargs(kwargs),projs=x.projs) for x,kwargs in zip(self.x,list_kwargs)]
        return self

    def noview(self):
        """Reset view."""
        self._kwargs_view = None
        return self

    @property
    def kwview(self):
        """Current view selections."""
        return self._kwargs_view or [{} for axis in range(NDIM)]

    @property
    def x(self):
        """Return data vectors for first and second dimensions."""
        return self._x

    @property
    def cov(self):
        """Return covariance matrix."""
        return self._cov

    @property
    def shape(self):
        """Return shape, i.e. tuple of length along the two dimensions."""
        return tuple(x.size for x in self._x)

    def get_x(self, concatenate=False, *args, **kwargs):
        """Return x-coordinates of the data vectors for the two dimensions."""
        indices = self.get_index(*args,concatenate=False,**kwargs)
        toret = []
        for x,index in zip(self._x,indices):
            isscalar = not isinstance(index,list)
            if isscalar: index = [index]
            x = [x.get_x(proj=proj)[index_] for proj,index_,_ in index]
            if concatenate: x = np.concatenate(x)
            elif isscalar: x = x[0]
            toret.append(x)
        return tuple(toret)

    def get_y(self, concatenate=True, *args, **kwargs):
        """Return mean y-coordinates for the two dimensions."""
        indices = self.get_index(*args,concatenate=False,**kwargs)
        toret = []
        for x,index in zip(self._x,indices):
            isscalar = not isinstance(index,list)
            if isscalar: index = [index]
            x = [x.get_y(proj=proj)[index_] for proj,index_,_ in index]
            if concatenate: x = np.concatenate(x)
            elif isscalar: x = x[0]
            toret.append(x)
        return tuple(toret)

    def set_new_edges(self, proj, *args, **kwargs):
        """
        Set new edges for projection of name ``proj`` (list or single projection name).
        See :meth:`BinnedStatistic.set_new_edges` for other arguments.
        """
        if not isinstance(proj,tuple):
            proj = (proj,)*NDIM
        matrices = []
        for x_,proj_ in zip(self.x,proj):
            x_.set_new_edges(proj,*args,**kwargs)
            matrices.append(x_._matrix_new_edges(proj,*args,flatten=True,**kwargs))
        self._cov = matrices[0].dot(cov).dot(matrices[1].T)

    def rebin(self, proj, *args, **kwargs):
        """
        Rebin projection of name ``proj`` (list or single projection name).
        See :meth:`BinnedStatistic.rebin` for other arguments.
        """
        if not isinstance(proj,tuple):
            proj = (proj,)*NDIM
        matrices = []
        for x_,proj_ in zip(self.x,proj):
            x_.rebin(proj,*args,**kwargs)
            matrices.append(x_._matrix_rebin(proj,*args,flatten=True,**kwargs))
        self._cov = matrices[0].dot(cov).dot(matrices[1].T)

    def get_std(self, *args, **kwargs):
        """Return standard deviation."""
        return np.diag(self.get_cov(*args,**kwargs))**0.5

    def get_cov(self, *args, **kwargs):
        """Return covariance."""
        return self._cov[np.ix_(*self.get_index(*args,**kwargs))]

    def get_invcov(self, *args, block=True, inv=np.linalg.inv, **kwargs):
        """
        Return inverse covariance.

        Parameters
        ----------
        block : bool, default=False
            Invert matrix per block.

        inv : callable, default=np.linalg.inv
            Function that inverts input matrix.

        args : list
            Arguments for :attr:`get_cov`.

        kwargs : list
            Arguments for :attr:`get_cov`.
        """
        if block:
            indices = self.get_index(concatenate=False,*args,**kwargs)
            indices = [[index] if not isinstance(index,list) else index for index in indices]
            cov = [[self._cov[np.ix_(ind1[1] + ind1[2],ind2[1] + ind2[2])] for ind2 in indices[-1]] for ind1 in indices[0]]
            return utils.blockinv(cov,inv=inv)
        return utils.inv(self.get_cov(*args,**kwargs))

    def get_corrcoef(self, *args, **kwargs):
        """Return correlation matrix."""
        return utils.cov_to_corrcoef(self.get_cov(*args,**kwargs))

    def __getstate__(self):
        """Return this class state dictionary."""
        state = {}
        for key in ['_cov','attrs']:
            state[key] = getattr(self,key)
        state['_x'] = [x.__getstate__() for x in self._x]
        state['_kwargs_view'] = _getstate_index_kwargs(self._kwargs_view) if self._kwargs_view is not None else None
        state['__class__'] = '.'.join((self.__class__.__module__,self.__class__.__name__))
        return state

    def __setstate__(self, state):
        """Set this class state dictionary."""
        super(CovarianceMatrix,self).__setstate__(state)
        self._x = [DataVector.from_state(x) for x in self._x]
        self._kwargs_view = _setstate_index_kwargs(state['_kwargs_view']) if state['_kwargs_view'] is not None else None

    @classmethod
    def load_auto(cls, filename, *args, **kwargs):
        """
        Load covariance matrix.

        Note
        ----
        Returned covariance matrix, if saved with a :class:`CovarianceMatrix`-inherited class, will be an instance of that class.

        Parameters
        ----------
        filename : string
            File name of covariance matrix.
            If ends with '.txt', calls :meth:`load_txt`
            Else (numpy binary format), calls :meth:`load`

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
        Write covariance matrix to disk.

        Parameters
        ----------
        filename : string
            File name of covariance matrix.
            If ends with '.txt', calls :meth:`save_txt`
            Else (numpy binary format), calls :meth:`save`

        args : list
            Arguments for save function.

        kwargs : dict
            Other arguments for save function.
        """
        if os.path.splitext(filename)[-1] == '.txt':
            return self.save_txt(filename,*args,**kwargs)
        return self.save(filename)

    @classmethod
    def load_txt(cls, filename, data=None, comments='#', usecols=None, skip_rows=0, max_rows=None, mapping_header=None, columns=None, mapping_proj=None, attrs=None, **kwargs):
        """
        Load :class:`CovarianceMatrix` from disk.

        Note
        ----
        If previously saved using :meth:`save_txt`, loading the :class:`CovarianceMatrix` only requires ``filename``.
        In this case, the returned instance will be of the class that was used to create it (e.g. :class:`MockCovarianceMatrix`)
        - not necessarily :class:`CovarianceMatrix`.

        Parameters
        ----------
        filename : string
            File name to read in.

        data : tuple, DataVector
            Data vector(s) matching the two dimensions of the covariance matrix.
            If provided, 'x' columns of the covariance file are expected to be indices of the data vector.

        comments : string, default='#'
            Characters used to indicate the start of a comment.

        usecols : list, default=None
            Which columns to read, with 0 being the first. If ``None``, reads all columns. Last column must be the covariance value.

        skip_rows : int, default=0
            Skip the first ``skip_rows`` lines, including comments.

        max_rows : int, default=None
            Read ``max_rows lines`` of content after ``skip_rows`` lines. The default is to read all the lines.

        mapping_header : dict, default=None
            Dictionary holding key:regex mapping or (regex, type) to provide the type.
            The corresponding values, read in the header, will be saved in the :attr:`attrs` dictionary.

        columns : list, default=None
            Column names corresponding to ``usecols``. Can be a tuple of column lists for two different data vectors.
            Columns 'x' and 'y' are used as x- and y-coordinates for each data vector.

        mapping_proj : dict, list, default=None
            List of projection names (considered of the same size), or dictionary holding a mapping from projection specifier (e.g. 'ell_0')
            to the number of points for this projection (e.g. ``{'ell_0':10, 'ell_2':4}`` for a matrix of total size 14 x 14).

        attrs : dict, default=None
            Attributes to save in the :attr:`attrs` dictionary.

        kwargs : dict
            Other arguments for :meth:`BinnedProjection.__init__`.

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
                    if DataVector.read_title_label(line[len(comments):]):
                        iline_seps.append(iline+1)
            iline_seps.append(len(file))
        else:
            cls.log_info('Not in {} standard format.'.format(cls.__class__.__name__),rank=0)
            iline_seps = [0,len(file)]
        attrs = (attrs or {}).copy()

        def str_to_float(e):
            return float(e)

        def str_to_int(e):
            return int(e)

        kwargs_view = None
        cov = []
        if self_format:
            header = cls.read_header_txt(file[iline_seps[0]:iline_seps[1]],comments=comments,mapping_header=mapping_header)
            kwargs_view = header.pop('kwargs_view',None)
            attrs = utils.dict_nonedefault(attrs,**header)
            start,stop = iline_seps[:2]
            for line in file[start:stop]:
                if line.startswith(comments): continue
                line = line.strip()
                cov.append(str_to_float(line))
            x = []
            iline_seps = iline_seps[1:]
            for start,stop in zip(iline_seps[:2],iline_seps[1:]):
                x.append(DataVector.load_txt(file[start:stop],comments=comments))
            cov = np.array(cov).reshape([x_.size for x_ in x])
        else:
            start,stop = iline_seps[:2]
            if usecols is None:
                for line in file[start:stop]:
                    if line.startswith(comments): continue
                    usecols = range(len(line.strip().split()))
                    break
                if usecols is None:
                    raise ValueError('No columns found in file')
            if columns is None:
                if len(usecols) == 1:
                    columns = []
                if len(usecols) == 3:
                    columns = ['x']
                else:
                    raise ValueError('Specify the column names (but the last one - the covariance), "x" for x-coordinates')
            if not isinstance(columns,tuple):
                columns = (columns,)*2
            list_data = [{col:[] for col in columns_} for columns_ in columns]
            allcolumns = sum(columns,[])
            list_cov = []
            totalsize = 0
            for line in file[start:stop]:
                if line.startswith(comments): continue
                totalsize += 1
            if isinstance(mapping_proj,list):
                mapping_proj = {proj:int(totalsize**0.5)//len(mapping_proj) for proj in mapping_proj}
            if mapping_proj is None:
                mapping_proj = {None:int(totalsize**0.5)}
            totalsize = sum(mapping_proj.values())
            iline = 0
            for line in file[start:stop]:
                if line.startswith(comments): continue
                line = line.strip().split()
                for icol in usecols:
                    value = line[icol]
                    if icol == len(usecols) - 1:
                        x_ = str_to_float(value)
                        list_cov.append(x_)
                    else:
                        col = allcolumns[icol]
                        idata = int(icol >= len(columns[0]))
                        dcol = list_data[idata][col]
                        toappend = (idata == 1 and iline < totalsize) or (idata == 0 and iline % totalsize == 0) # first column
                        #if toappend: print(idata,iline,totalsize)
                        if col == 'x':
                            if data is not None:
                                x_ = str_to_int(value)
                            else:
                                x_ = str_to_float(value)
                            if toappend:
                                dcol.append(x_)
                        else:
                            x_ = str_to_float(value)
                            if toappend:
                                dcol.append(x_)
                iline += 1
            cov = np.empty((totalsize,)*2,dtype='f8')
            cov.flat[...] = list_cov
            if data is not None:
                if not isinstance(data,tuple):
                    data = (data,data)
                x,argsort = [],[]
                for data_,index_ in zip(data,list_data):
                    ix_ = np.asarray(index_['x'])
                    argsort_ = np.argsort(ix_)
                    ix_ = ix_[argsort_]
                    x_ = data_.copy()
                    start = 0
                    for proj,indexproj in x_.get_index():
                        dataproj = x_.get(proj)
                        tmpindex = np.arange(dataproj.size)
                        stop = start + dataproj.size
                        tmpindex = tmpindex[indexproj][ix_[(ix_ >= start) & (ix_ < stop)] - start]
                        start = stop
                        x_.set(dataproj[tmpindex])
                    x.append(x_)
                    argsort.append(argsort_)
                cov = cov[np.ix_(*argsort)]
            else:
                x = []
                for data_,columns_ in zip(list_data,columns):
                    binnedprojs = []
                    start = 0
                    for proj in mapping_proj:
                        stop = start + mapping_proj[proj]
                        #print(proj,len(data_[columns_[0]]),data_[columns_[0]])
                        binnedprojs.append(BinnedProjection({col:data_[col][start:stop] for col in columns_},proj=proj,**kwargs))
                        start = stop
                    x.append(DataVector(binnedprojs))

        new = cls.__new__(cls)
        CovarianceMatrix.__init__(new,cov,first=x[0],second=x[1],attrs=attrs)
        #print(x[0].size,x[1].size,cov.shape)
        if kwargs_view is not None: new._kwargs_view = _setstate_index_kwargs(kwargs_view)
        return new

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
            header.append('{}#{} = {}'.format(comments,'kwview',json.dumps(_getstate_index_kwargs(self._kwargs_view))))
        return header

    @savefile
    def save_txt(self, filename, comments='#', fmt='.18e', ignore_json_errors=True):
        lines = self.get_header_txt(comments=comments,ignore_json_errors=ignore_json_errors)

        for ix1 in range(self._x[0].size):
            for ix2 in range(self._x[1].size):
                lines.append('{:{fmt}}'.format(self._cov[ix1,ix2],fmt=fmt))

        for x in self._x:
            lines += x.save_txt(filename=None,comments=comments,fmt=fmt,ignore_json_errors=ignore_json_errors)

        if filename is not None:
            if self.is_mpi_root():
                with open(filename,'w') as file:
                    for line in lines:
                        file.write(line + '\n')
        else:
            return lines

    def plot(self, style='corr', data_styles=None, **kwargs_style):
        """Plot covariance matrix. See :class:`plotting.MatrixPlotStyle`."""
        from .plotting import MatrixPlotStyle
        style = MatrixPlotStyle(style,covariance=self,data_styles=data_styles,**kwargs_style)
        style.plot()


CovarianceMatrix.get_title_label = get_title_label
CovarianceMatrix.read_title_label = read_title_label
CovarianceMatrix.read_header_txt = read_header_txt


class MockCovarianceMatrix(CovarianceMatrix):

    """Class representing a covariance matrix built from mocks."""

    logger = logging.getLogger('MockCovarianceMatrix')

    @classmethod
    def from_data(cls, *list_data):
        """Estimate covariance matrix from list of data vectors."""
        list_dataproj,list_y = {},[]
        for ldata in zip(*list_data):
            data = DataVector.concatenate(*ldata)
            for proj in data.projs:
                dataproj = data.get(proj)
                if proj not in list_dataproj:
                    list_dataproj[proj] = {col:[] for col in dataproj.columns}
                for col in dataproj.columns:
                    list_dataproj[proj][col].append(dataproj[col])
            list_y.append(data.get_y())
        data = data.copy(copy_proj=True)
        for proj in list_dataproj:
            dataprojmean = data.get(proj)
            dataprojmean.data = {col: np.mean(list_dataproj[proj][col],axis=0) for col in list_dataproj[proj]}
            data.set(dataprojmean)
        covariance = np.cov(np.array(list_y).T,ddof=1)
        return cls(covariance=covariance,first=data,attrs={'nobs':len(list_y)})

    @classmethod
    def from_files(cls, reader, *filenames, **kwargs):
        """
        Estimate covariance matrix from list of data vector files.

        Parameters
        ----------
        reader : callable
            Function that takes in a file name (string) and ``kwargs`` as input and returns a :class:`DataVector` instance.

        filenames : list
            List of data vector file names.

        kwargs : dict
            Arguments for ``reader``.

        Returns
        -------
        covariance : MockCovarianceMatrix
        """
        list_data = ((reader(fn, **kwargs) for fn in fns) for fns in filenames)
        return cls.from_data(*list_data)
