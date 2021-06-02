import logging
import json

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

from .data_vector import DataVector
from cosmopipe.lib.utils import BaseClass
from cosmopipe.lib import utils


class CovarianceMatrix(DataVector):

    logger = logging.getLogger('CovarianceMatrix')

    def __init__(self, covariance, x=None, mean=None, proj=None, mapping_proj=None, **attrs):

        if isinstance(covariance,self.__class__):
            self.__dict__.update(covariance.__dict__)
            self.attrs.update(attrs)
            return

        if not isinstance(x,tuple):
            x = (x,)*covariance.ndim
        if not isinstance(mean,tuple):
            mean = (mean,)*covariance.ndim
        if not isinstance(proj,tuple):
            proj = (proj,)*covariance.ndim
        if not isinstance(mapping_proj,tuple):
            mapping_proj = (mapping_proj,)*covariance.ndim
        self._x = list(DataVector(x=x_,y=mean_,proj=proj_,mapping_proj=mapping_proj_)\
                                for x_,mean_,proj_,mapping_proj_ in zip(x,mean,proj,mapping_proj))
        self._covariance = covariance
        self.attrs = attrs

    def get_index(self, axes=None, **kwargs):
        if axes is None:
            axes = range(self.ndim)
        if np.isscalar(axes):
            axes = [axes]
        for key,val in kwargs.items():
            if not isinstance(val,tuple):
                kwargs[key] = (val,)*len(axes)
        index = [None for axis in range(self.ndim)]
        for axis in axes:
            index[axis] = self._x[axis].get_index(**{key:val[axis] for key,val in kwargs.items()})
        return index

    def copy(self):
        new = super(CovarianceMatrix,self).copy()
        for axis,x in enumerate(self._x):
            new._x[axis] = x.copy()
        return new

    def view(self, **kwargs):
        new = self.copy()
        masks = self.get_index(**kwargs)
        for iaxis,x in enumerate(new._x):
            x._index_view = masks[iaxis]
        return new

    def noview(self):
        new = self.copy()
        for iaxis,x in enumerate(new._x):
            new._x[iaxis] = x.noview()

    def __getitem__(self, mask):
        new = self.copy()
        if not isinstance(mask,tuple):
            mask = (mask,)*self.ndim
        for ix,m in enumerate(mask):
            self._x[ix] = self._x[ix][m]
        new._covariance = new._covariance[np.ix_(*mask)]
        return new

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def cov(self):
        return self._covariance

    def get_x(self, **kwargs):
        """Return x-coordinates of the data vector."""
        indices = self.get_index(**kwargs)
        return [x.x[index] for x,index in zip(self._x,indices)]

    def get_y(self, **kwargs):
        """Return mean data vector."""
        indices = self.get_index(**kwargs)
        return [x.y[index] for x,index in zip(self._x,indices)]

    def get_proj(self, **kwargs):
        indices = self.get_index(**kwargs)
        return [x.get_proj()[index] for x,index in zip(self._x,indices)]

    def get_std(self, **kwargs):
        return np.diag(self.get_cov(**kwargs))**0.5

    def get_cov(self, **kwargs):
        return self._covariance[np.ix_(*self.get_index(**kwargs))]

    def get_invcov(self, block=True, inv=np.linalg.inv, **kwargs):
        if block:
            indices = self.get_index(concat=False,**kwargs)
            cov = [[self._covariance[np.ix_(ind1,ind2)] for ind2 in indices[-1]] for ind1 in indices[0]]
            return utils.blockinv(cov,inv=inv)
        return utils.inv(self.get_cov(**kwargs))

    def get_corrcoef(self, **kwargs):
        return utils.cov_to_corrcoef(self.get_cov(**kwargs))

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def shape(self):
        return tuple(x.size for x in self._x)

    def __getstate__(self):
        state = BaseClass.__getstate__(self)
        for key in ['_covariance']:
            state[key] = getattr(self,key)
        state['_x'] = [x.__getstate__() for x in self._x]
        return state

    def __setstate__(self,state):
        BaseClass.__setstate__(self,state)
        self._x = [DataVector.from_state(x) for x in self._x]

    @classmethod
    def load_auto(cls, filename, *args, **kwargs):
        if os.path.splitext(filename)[-1] == '.txt':
            return cls.load_txt(filename,*args,**kwargs)
        return cls.load(filename,*args,**kwargs)

    def save_auto(self, filename, *args, **kwargs):
        if os.path.splitext(filename)[-1] == '.txt':
            return self.save_txt(filename,*args,**kwargs)
        return self.save(filename)

    @classmethod
    def load_txt(cls, filename, data=None, mapping_header=None, xdim=None, comments='#', usecols=None, skip_rows=0, max_rows=None, **attrs):
        cls.log_info('Loading {}.'.format(filename),rank=0)

        with open(filename,'r') as file:
            header = cls.read_header_txt(file,mapping_header=mapping_header,comments=comments)

        attrs = {**header,**attrs}
        col_proj = isinstance(attrs.get('proj',None),bool) and attrs['proj']
        x,cov,mapping = [[],[]],[],[]
        proj,projx = [[],[]],[[],[]]

        def str_to_y(e):
            return float(e)

        if data is not None:

            def str_to_x(row):
                return [int(e) for e in row]

        else:

            def str_to_x(row):
                return [float(e) for e in row]

        with open(filename,'r') as file:
            for iline,line in enumerate(file):
                if iline < skip_rows: continue
                if max_rows is not None and iline >= skip_rows + max_rows: break
                if line.startswith(comments): continue
                row = line.split()
                if usecols is None:
                    usecols = range(len(row))
                ixl = 2 if col_proj else 0
                if xdim is None:
                    nx = len(usecols) - 1 - ixl
                    if nx % 2 == 0:
                        xdim = (nx//2,nx//2)
                    else:
                        raise ValueError('x vectors do not have the same dimensions; please provide xdim')
                slx = (slice(ixl,ixl+xdim[0]),slice(ixl+xdim[0],ixl+xdim[0]+xdim[1]))
                row = [row[icol] for icol in usecols]
                mapping_ = []
                if col_proj:
                    for i in range(2):
                        x_ = str_to_x(row[slx[i]])
                        projx_ = tuple([row[i]] + x_)
                        if projx_ not in projx[i]:
                            projx[i].append(projx_)
                            proj[i].append(row[i])
                            x[i].append(x_)
                        mapping_.append(projx[i].index(projx_))
                else:
                    for i in range(2):
                        x_ = str_to_x(row[slx[i]])
                        if x_ not in x[i]:
                            x[i].append(x_)
                        mapping_.append(x[i].index(x_))
                mapping.append(mapping_)
                cov.append(str_to_y(row[-1]))

        mapping = np.array(mapping).T
        mcov = np.full(mapping.max(axis=-1)+1,np.nan)
        mcov[tuple(mapping)] = cov

        x = tuple(np.squeeze(x_) for x_ in x)
        if col_proj:
            attrs['proj'] = tuple(np.array(p) for p in proj)

        attrs.setdefault('filename',filename)
        if data is not None:
            x = tuple(data[ix] for ix in x)

        return cls(mcov,x=x,**attrs)

    @utils.savefile
    def save_txt(self, filename, comments='#', fmt='.18e'):
        if self.is_mpi_root():
            with open(filename,'w') as file:
                for key,val in self.attrs.items():
                    file.write('{}{} = {}\n'.format(comments,key,json.dumps(val)))
                if self._x[0].has_proj():
                    file.write('{}projection = {}\n'.format(comments,json.dumps(True)))
                for ix1,x1 in enumerate(self._x[0]._x):
                    for ix2,x2 in enumerate(self._x[1]._x):
                        if self._x[0].has_proj():
                            file.write('{} {} {:{fmt}} {:{fmt}} {:{fmt}}\n'.format(self._x[0]._proj[ix1],self._x[1]._proj[ix2],x1,x2,\
                                                                                    self._covariance[ix1,ix2],fmt=fmt))
                        else:
                            file.write('{:{fmt}} {:{fmt}} {:{fmt}}\n'.format(x1,x2,self._cov[ix1,ix2],fmt=fmt))

    def plot(self, style='corr', data_styles=None, **kwargs_style):
        from .plotting import DataPlotStyle
        style = DataPlotStyle(style,data_styles=data_styles,**kwargs_style)
        style.plot(covariance=self)



class MockCovarianceMatrix(CovarianceMatrix):

    logger = logging.getLogger('MockCovarianceMatrix')

    @classmethod
    def from_data(cls, list_data):
        list_x,list_y = [],[]
        for data in list_data:
            list_x.append(data.get_x())
            list_y.append(data.get_y())
        x = np.mean(list_x,axis=0)
        mean = np.mean(list_y,axis=0)
        covariance = np.cov(np.array(list_y).T,ddof=1)
        x = DataVector(x=x,y=mean,proj=data._proj)
        return cls(covariance=covariance,x=x,mean=mean,nobs=len(list_y))

    @classmethod
    def from_files(cls, reader, filenames, **kwargs):
        filenames = filenames or []
        list_data = (reader(filename, **kwargs) for filename in filenames)
        return cls.from_data(list_data)
