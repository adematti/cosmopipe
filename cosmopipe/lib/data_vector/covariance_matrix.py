import os
import logging
import json

import numpy as np

from cosmopipe.lib.utils import BaseClass, savefile
from cosmopipe.lib import utils
from .data_vector import DataVector
from .binned_statistic import BinnedProjection, _title_template


NDIM = 2


def _format_index_kwargs(kwargs, axis=None):
    if axis is None:
        axis = range(NDIM)
    if np.ndim(axis) == 0:
        axis = [axis]
    toret = [{} for axis in range(NDIM)]
    for key,value in kwargs.items():
        if not isinstance(value,tuple):
            value = (value,)*len(axis)
        for iax,val in zip(axis,value):
            toret[iax][key] = val
    return toret


class CovarianceMatrix(BaseClass):

    logger = logging.getLogger('CovarianceMatrix')

    _default_mapping_header = {'kwargs_view':'.*?kwview = (.*)$'}

    def __init__(self, covariance, x, x2=None, attrs=None):

        if isinstance(covariance,self.__class__):
            self.__dict__.update(covariance.__dict__)
            return

        self.attrs = attrs or {}
        self._cov = covariance
        # copy to avoid issues related to changes in _x but not in _cov
        self._x = [DataVector(x).copy()]*NDIM
        if x2 is not None: self._x[1] = DataVector(x2).copy()

    def get_index(self, concatenate=True, **kwargs):
        kwargs = _format_index_kwargs(kwargs)
        index = [None for axis in range(NDIM)]
        for axis in range(NDIM):
            index[axis] = self._x[axis].get_index(**{key:value for key,value in kwargs[axis].items()},glob=True,concatenate=concatenate)
        return index

    def __copy__(self):
        new = super(CovarianceMatrix,self).__copy__()
        for name in ['attrs']:
            setattr(new,name,getattr(new,name).copy())
        new._x = [x.copy() for x in self._x]
        return new

    def view(self, **kwargs):
        kwargs = _format_index_kwargs(kwargs)
        for axis,x in enumerate(self._x):
            x.view(**kwargs[axis])
        return self

    def noview(self):
        for x in self._x:
            x.noview()
        return self

    def __getitem__(self, mask):
        new = self.copy()
        if not isinstance(mask,tuple):
            mask = (mask,)*NDIM
        for ix,m in enumerate(mask):
            self._x[ix] = self._x[ix][m]
        new._cov = new._cov[np.ix_(*mask)]
        return new

    @property
    def x(self):
        return self._x

    @property
    def cov(self):
        return self._cov

    @property
    def shape(self):
        return tuple(x.size for x in self._x)

    def get_x(self, **kwargs):
        """Return x-coordinates of the data vector."""
        kwargs = _format_index_kwargs(kwargs)
        return tuple(x.get_x(**kw) for x,kw in zip(self._x,kwargs))

    def get_y(self, **kwargs):
        """Return mean data vector."""
        kwargs = _format_index_kwargs(kwargs)
        return tuple(x.get_y(**kw) for x,kw in zip(self._x,kwargs))

    def set_new_edges(self, proj, *args, **kwargs):
        if not isinstance(proj,tuple):
            proj = (proj,)*NDIM
        matrices = []
        for x_,proj_ in zip(self.x,proj):
            x_.set_new_edges(proj,*args,**kwargs)
            matrices.append(x_._matrix_new_edges(proj,*args,flatten=True,**kwargs))
        self._cov = matrices[0].dot(cov).dot(matrices[1].T)

    def rebin(self, proj, *args, **kwargs):
        if not isinstance(proj,tuple):
            proj = (proj,)*NDIM
        matrices = []
        for x_,proj_ in zip(self.x,proj):
            x_.rebin(proj,*args,**kwargs)
            matrices.append(x_._matrix_rebin(proj,*args,flatten=True,**kwargs))
        self._cov = matrices[0].dot(cov).dot(matrices[1].T)

    def get_std(self, **kwargs):
        return np.diag(self.get_cov(**kwargs))**0.5

    def get_cov(self, **kwargs):
        return self._cov[np.ix_(*self.get_index(**kwargs))]

    def get_invcov(self, block=True, inv=np.linalg.inv, **kwargs):
        if block:
            indices = self.get_index(concatenate=False,**kwargs)
            cov = [[self._cov[np.ix_(ind1,ind2)] for ind2 in indices[-1]] for ind1 in indices[0]]
            return utils.blockinv(cov,inv=inv)
        return utils.inv(self.get_cov(**kwargs))

    def get_corrcoef(self, **kwargs):
        return utils.cov_to_corrcoef(self.get_cov(**kwargs))

    def __getstate__(self):
        state = {}
        for key in ['_cov','attrs']:
            state[key] = getattr(self,key)
        state['_x'] = [x.__getstate__() for x in self._x]
        return state

    def __setstate__(self,state):
        super(CovarianceMatrix,self).__setstate__(state)
        self._x = [DataVector.from_state(x) for x in self._x]

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
    def load_txt(cls, filename, data=None, mapping_header=None, comments='#', usecols=None, columns=None, skip_rows=0, max_rows=None, attrs=None, **kwargs):
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
                if iline >= skip_rows and line.startswith(comments):
                    if line[len(comments):-1] == DataVector.get_title_label(): # -1 because last character is \n
                        iline_seps.append(iline)
            iline_seps.append(iline+1)
        else:
            iline_seps = [skip_rows,max_rows]
        attrs = (attrs or {}).copy()

        def str_to_float(e):
            return float(e)

        def str_to_int(e):
            return int(e)

        cov = []
        if self_format:
            header = cls.read_header_txt(file[iline_seps[0]:iline_seps[1]],comments=comments,mapping_header=mapping_header)
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
            list_cov, mapping = [], []
            for line in file[start:stop]:
                if line.startswith(comments): continue
                line = line.strip().split()
                mapping_ = []
                for icol in usecols:
                    value = line[icol]
                    if icol == len(usecols) - 1:
                        x_ = str_to_float(value)
                        list_cov.append(x_)
                    else:
                        col = allcolumns[icol]
                        idata = int(icol >= len(columns[0]))
                        if col == 'x':
                            if data is not None:
                                x_ = str_to_int(value)
                            else:
                                x_ = str_to_float(value)
                            if x_ not in list_data[idata][col]:
                                list_data[idata][col].append(x_)
                            mapping_.append(list_data[idata][col].index(x_))
                        else:
                            x_ = str_to_float(value)
                            if x_ not in list_data[idata][col]:
                                list_data[idata][col].append(x_)
                mapping.append(mapping_)
            mapping = np.array(mapping).T
            cov = np.full(mapping.max(axis=-1)+1,np.nan)
            cov[tuple(mapping)] = list_cov
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
                x = tuple(BinnedProjection({col:data_[col] for col in columns_},**kwargs) for data_,columns_ in zip(list_data,columns))

        return cls(cov,x=x[0],x2=x[1],attrs=attrs)

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
        from .plotting import MatrixPlotStyle
        style = MatrixPlotStyle(style,covariance=self,data_styles=data_styles,**kwargs_style)
        style.plot()


CovarianceMatrix.read_header_txt = DataVector.read_header_txt


class MockCovarianceMatrix(CovarianceMatrix):

    logger = logging.getLogger('MockCovarianceMatrix')

    @classmethod
    def from_data(cls, *list_data):
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
        for proj in list_dataproj:
            dataprojmean = data.get(proj)
            dataprojmean.data = {col: np.mean(list_dataproj[proj][col],axis=0) for col in list_dataproj[proj]}
            data.set(dataprojmean)
        covariance = np.cov(np.array(list_y).T,ddof=1)
        return cls(covariance=covariance,x=data,attrs={'nobs':len(list_y)})

    @classmethod
    def from_files(cls, reader, *filenames, **kwargs):
        list_data = ((reader(fn, **kwargs) for fn in fns) for fns in filenames)
        return cls.from_data(*list_data)
