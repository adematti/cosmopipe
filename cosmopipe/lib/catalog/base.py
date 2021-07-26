import os
import re
import functools
import logging

import numpy as np

from cosmopipe.lib.utils import ScatteredBaseClass
from cosmopipe.lib import utils, mpi


def vectorize_columns(func):
    @functools.wraps(func)
    def wrapper(self, column, **kwargs):
        if not self.__class__._multiple_columns(column):
            return func(self,column,**kwargs)
        toret = [func(self,col,**kwargs) for col in column]
        if all(t is None for t in toret): # in case not broadcast to all ranks
            return None
        return np.asarray(toret)
    return wrapper


class BaseCatalog(ScatteredBaseClass):

    logger = logging.getLogger('BaseCatalog')
    _broadcast_attrs = ['attrs','mpistate','mpiroot']

    @classmethod
    def _multiple_columns(cls, column):
        return isinstance(column,list)

    @mpi.MPIInit
    def __init__(self, data=None, columns=None, attrs=None):
        self.data = {}
        if columns is None:
            columns = list((data or {}).keys())
        if data is not None:
            for name in columns:
                self[name] = data[name]
        self.attrs = attrs or {}

    @mpi.MPIBroadcast
    def mpi_broadcast(self, onroot):
        isroot = self.is_mpi_root()
        for key in self._broadcast_attrs:
            setattr(self,key,self.mpicomm.bcast(getattr(onroot,key) if isroot else None,root=self.mpiroot))
        columns = self.mpicomm.bcast(list(onroot.data.keys()) if isroot else None,root=self.mpiroot)
        self.data = {}
        for col in columns:
            self[col] = mpi.broadcast_array(onroot[col] if isroot else None,root=self.mpiroot,mpicomm=self.mpicomm)

    @mpi.MPIScatter
    def mpi_scatter(self):
        isroot = self.is_mpi_root()
        for key in self._broadcast_attrs:
            setattr(self,key,self.mpicomm.bcast(getattr(self,key) if isroot else None,root=self.mpiroot))
        columns = self.columns()
        if not isroot:
            self.data = {}
        for col in columns:
            self[col] = mpi.scatter_array(self[col] if isroot else None,root=self.mpiroot,mpicomm=self.mpicomm)

    def mpi_send(self, dest, tag=42):
        for key in self._broadcast_attrs:
            self.mpicomm.send(getattr(self,key),dest=dest,tag=tag)
        columns = list(self.data.keys())
        self.mpicomm.send(columns,dest=dest,tag=tag)
        for col in columns:
            mpi.send_array(self[col],dest=dest,tag=tag,mpicomm=self.mpicomm)

    def mpi_recv(self, source, tag=42):
        for key in self._broadcast_attrs:
            setattr(self,key,self.mpicomm.recv(source=source,tag=tag))
        columns = self.mpicomm.recv(source=source,tag=tag)
        for col in columns:
            self[col] = mpi.recv_array(source=source,tag=tag,mpicomm=self.mpicomm)

    @mpi.MPIGather
    def mpi_gather(self):
        for col in self.columns():
            self[col] = mpi.gather_array(self[col],root=self.mpiroot,mpicomm=self.mpicomm)

    def __len__(self):
        keys = list(self.data.keys())
        if not keys or self[keys[0]] is None:
            return 0
        return len(self[keys[0]])

    @property
    def size(self):
        return len(self)

    @property
    def gsize(self):
        if self.is_mpi_scattered():
            return self.mpicomm.allreduce(len(self))
        isroot = self.is_mpi_root()
        return self.mpicomm.bcast(len(self) if isroot else None,root=self.mpiroot)

    def columns(self, include=None, exclude=None):
        toret = None

        if self.is_mpi_root():
            toret = allcols = list(self.data.keys())

            def toregex(name):
                return name.replace('.','\.').replace('*','(.*)')

            if include is not None:
                if not isinstance(include,(tuple,list)):
                    include = [include]
                toret = []
                for inc in include:
                    inc = toregex(inc)
                    for col in allcols:
                        if re.match(inc,str(col)):
                            toret.append(col)
                allcols = toret

            if exclude is not None:
                if not isinstance(exclude,(tuple,list)):
                    exclude = [exclude]
                toret = []
                for exc in exclude:
                    exc = toregex(exc)
                    for col in allcols:
                        if re.match(exc,str(col)) is None:
                            toret.append(col)

        return self.mpicomm.bcast(toret,root=self.mpiroot)

    def __contains__(self, column):
        return column in self.data

    def indices(self):
        sizes = self.mpicomm.allgather(len(self))
        sizes = [0] + np.cumsum(sizes[:1]).tolist()
        return sizes[self.mpicomm.rank] + np.arange(len(self))

    def zeros(self, dtype=np.float64):
        return np.zeros(len(self),dtype=dtype)

    def ones(self, dtype=np.float64):
        return np.ones(len(self),dtype=dtype)

    def full(self, fill_value, dtype=np.float64):
        return np.full(len(self),fill_value,dtype=dtype)

    def falses(self):
        return self.zeros(dtype=np.bool_)

    def trues(self):
        return self.ones(dtype=np.bool_)

    def nans(self):
        return self.ones()*np.nan

    def get(self, column, *args, **kwargs):
        has_default = False
        if args:
            if len(args) > 1:
                raise SyntaxError('Too many arguments!')
            has_default = True
            default = args[0]
        if kwargs:
            if len(kwargs) > 1:
                raise SyntaxError('Too many arguments!')
            has_default = True
            default = kwargs['default']
        if column not in self.data and has_default:
            return default
        return self.data[column]

    def set(self, column, item):
        self.data[column] = item

    def gget(self, column, root=None):
        if root is None: root = Ellipsis
        if self.is_mpi_scattered():
            return mpi.gather_array(self[column],root=Ellipsis,mpicomm=self.mpicomm)
        if self.is_mpi_gathered():
            if root is Ellipsis:
                return mpi.broadcast_array(self[column] if self.is_mpi_root() else None,root=self.mpiroot,mpicomm=self.mpicomm)
            elif root == self.mpiroot:
                if self.is_mpi_root(): return self[column]
                return
            # root != self.mpiroot
            if self.is_mpi_root(): mpi.send_array(self[column],dest=self.mpiroot,tag=42,mpicomm=self.mpicomm)
            if self.mpicomm.rank == root:
                return mpi.recv_array(source=self.mpiroot,tag=42,mpicomm=self.mpicomm)
            return
        # broadcast
        if root is Ellipsis or self.is_mpi_root():
            return self[column]

    def gslice(self, *args):
        sl = slice(*args)
        new = self.copy()
        for col in self.columns():
            self_value = self.gget(col,root=self.mpiroot)
            new[col] = None
            if self.is_mpi_root():
                new[col] = self_value[sl]
            if self.is_mpi_scattered():
                new[col] = mpi.scatter_array(new[col] if self.is_mpi_root() else None,root=self.mpiroot,mpicomm=self.mpicomm)
            elif self.is_mpi_broadcast():
                new[col] = mpi.broadcast_array(new[col] if self.is_mpi_root() else None,root=self.mpiroot,mpicomm=self.mpicomm)
        return new

    def to_array(self, columns=None, struct=True):
        if self.is_mpi_gathered() and not self.is_mpi_root():
            return None
        if columns is None:
            columns = self.columns()
        if struct:
            toret = np.empty(self.size,dtype=[(col,self[col].dtype,self[col].shape[1:]) for col in columns])
            for col in columns: toret[col] = self[col]
            return toret
        return np.array([self[col] for col in columns])

    @classmethod
    @mpi.CurrentMPIComm.enable
    def from_array(cls, array, columns=None, mpiroot=0, mpistate=mpi.CurrentMPIState.SCATTERED, mpicomm=None, **kwargs):
        isstruct = None
        if mpicomm.rank == mpiroot:
            isstruct = array.dtype.names is not None
            if isstruct:
                if columns is None: columns = array.dtype.names
        isstruct = mpicomm.bcast(isstruct,root=mpiroot)
        columns = mpicomm.bcast(columns,root=mpiroot)
        new = cls(data=dict.fromkeys(columns),mpiroot=mpiroot,mpistate=mpistate,mpicomm=mpicomm,**kwargs)
        if new.is_mpi_gathered() and not new.is_mpi_root():
            return new
        if isstruct:
            new.data = {col:array[col] for col in columns}
        else:
            new.data = {col:arr for col,arr in zip(columns,array)}
        return new

    def copy(self, columns=None):
        new = super(BaseCatalog,self).__copy__()
        if columns is None: columns = self.columns()
        new.data = {col:self[col] if col in self else None for col in columns}
        return new

    def __getstate__(self):
        """Return this class state dictionary."""
        data = {str(name):col for name,col in self.data.items()}
        return {'data':data,'attrs':self.attrs}

    def __setstate__(self, state):
        """Set the class state dictionary."""
        self.data = state['data'].copy()
        self.attrs = state['attrs']

    def __getitem__(self, name):
        if isinstance(name,str):
            return self.get(name)
        new = self.copy()
        new.data = {col:self[col][name] for col in self.data}
        return new

    def __setitem__(self, name, item):
        if isinstance(name,str):
            return self.set(name,item)
        for col in self.data:
            self[col][name] = item

    def __delitem__(self, name):
        del self.data[name]

    def __repr__(self):
        return '{}(size={:d}, columns={})'.format(self.__class__.__name__,self.gsize,self.columns())

    def extend(self, other):
        self_columns = self.columns()
        other_columns = other.columns()
        assert self.mpicomm is other.mpicomm

        if self_columns and other_columns and set(other_columns) != set(self_columns):
            raise ValueError('Cannot extend samples as columns do not match: {} != {}.'.format(other_columns,self_columns))

        for col in other_columns:
            if col not in self_columns:
                self_value = None
            else:
                self_value = self.gget(col,root=self.mpiroot)
            other_value = other.gget(col,root=self.mpiroot)
            self[col] = None
            if self.is_mpi_root():
                if self_value is not None:
                    self[col] = np.concatenate([self_value,other_value],axis=0)
                else:
                    self[col] = other_value.copy()
            if self.is_mpi_scattered():
                self[col] = mpi.scatter_array(self[col] if self.is_mpi_root() else None,root=self.mpiroot,mpicomm=self.mpicomm)
            elif self.is_mpi_broadcast():
                self[col] = mpi.broadcast_array(self[col] if self.is_mpi_root() else None,root=self.mpiroot,mpicomm=self.mpicomm)
        return self

    def __eq__(self, other):
        if not isinstance(other,self.__class__):
            return False
        self_columns = self.columns()
        other_columns = other.columns()
        if set(other_columns) != set(self_columns):
            return False
        assert self.mpicomm is other.mpicomm
        toret = True
        for col in self_columns:
            self_value = self.gget(col,root=self.mpiroot)
            other_value = other.gget(col,root=self.mpiroot)
            if self.is_mpi_root():
                if not np.all(self_value == other_value):
                    toret = False
                    break
        return self.mpicomm.bcast(toret,root=self.mpiroot)

    def save_auto(self, filename, *args, **kwargs):
        if os.path.splitext(filename)[-1] == '.fits':
            return self.save_fits(filename,*args,**kwargs)
        return self.save(filename)

    @classmethod
    def load_auto(cls, filename, *args, **kwargs):
        if os.path.splitext(filename)[-1] == '.fits':
            return cls.load_fits(filename,*args,**kwargs)
        return cls.load(filename,**kwargs)

    @classmethod
    def load(cls, *args, **kwargs):
        """Load class from disk."""
        self = super(BaseCatalog,cls).load(*args,**kwargs)
        for key in self._broadcast_attrs:
            setattr(self,key,self.mpicomm.bcast(getattr(self,key) if self.is_mpi_root() else None,root=self.mpiroot))
        for col in self.columns():
            if col not in self: self.data[col] = None
        return self

    @classmethod
    @mpi.CurrentMPIComm.enable
    def load_fits(cls, filename,  columns=None, ext=None, mpiroot=0, mpistate=mpi.CurrentMPIState.SCATTERED, mpicomm=None):
        """Load class from disk."""
        cls.log_info('Loading {}.'.format(filename),rank=0)
        import fitsio
        # Stolen from https://github.com/bccp/nbodykit/blob/master/nbodykit/io/fits.py
        msg = 'Input FITS file {}'.format(filename)
        with fitsio.FITS(filename) as ff:
            if ext is None:
                for i, hdu in enumerate(ff):
                    if hdu.has_data():
                        ext = i
                        break
                if ext is None:
                    raise IOError('{} has no binary table to read'.format(msg))
            else:
                if isinstance(ext,str):
                    if ext not in ff:
                        raise IOError('{} does not contain extension with name {}'.format(msg,ext))
                elif ext >= len(ff):
                    raise IOError('{} extension {} is not valid'.format(msg,ext))
            ff = ff[ext]
            # make sure we crash if data is wrong or missing
            if not ff.has_data() or ff.get_exttype() == 'IMAGE_HDU':
                raise ValueError('{} extension {} is not a readable binary table'.format(msg,ext))
            size = ff.get_nrows()
            start = mpicomm.rank * size // mpicomm.size
            stop = (mpicomm.rank + 1) * size // mpicomm.size
            new = ff.read(ext=ext,columns=columns,rows=range(start,stop))
            header = ff.read_header()
            header.clean()
            attrs = dict(header)
            attrs['fitshdr'] = header
            new = cls.from_array(new,mpiroot=mpiroot,mpistate=mpi.CurrentMPIState.SCATTERED,mpicomm=mpicomm,attrs=attrs)
        new = new.mpi_to_state(mpistate)
        return new

    @utils.savefile
    def save_fits(self, filename):
        """Save as fits file. Should be possible to change fitsio to write by chunks?."""
        import fitsio
        array = self.to_array(struct=True)
        if self.is_mpi_scattered():
            array = mpi.gather_array(array,root=self.mpiroot,mpicomm=self.mpicomm)
        if self.is_mpi_root():
            fitsio.write(filename,array,header=self.attrs.get('fitshdr',None),clobber=True)

    @classmethod
    def from_nbodykit(cls, catalog, columns=None):
        if columns is None: columns = catalog.columns
        data = {col: catalog[col].compute() for col in columns}
        return cls(data,mpicomm=catalog.comm,mpistate='scattered',mpiroot=0,attrs=catalog.attrs)

    def to_nbodykit(self, columns=None):
        if columns is None: columns = self.columns()
        if not self.is_mpi_scattered():
            self = self.copy()
            self.mpi_scatter()
        source = {col:self[col] for col in columns}
        from nbodykit.lab import ArrayCatalog
        return ArrayCatalog(source,**self.attrs)

    @vectorize_columns
    def sum(self, column):
        # NOTE: not weighted!!!
        if self.is_mpi_scattered():
            return mpi.sum_array(self[column],mpicomm=self.mpicomm)
        isroot = self.is_mpi_root()
        return self.mpicomm.bcast(np.sum(self[column]) if isroot else None,root=self.mpiroot)

    @vectorize_columns
    def average(self, column, weights=None):
        if self.is_mpi_scattered():
            return mpi.average_array(self[column],weights=weights,mpicomm=self.mpicomm)
        isroot = self.is_mpi_root()
        return self.mpicomm.bcast(np.average(self[column],weights=weights) if isroot else None,root=self.mpiroot)

    @vectorize_columns
    def mean(self, column):
        return self.average(column)

    @vectorize_columns
    def median(self, column):
        return self.quantile(column,q=0.5)

    @vectorize_columns
    def minimum(self, column):
        if self.is_mpi_scattered():
            return mpi.min_array(self[column],mpicomm=self.mpicomm)
        isroot = self.is_mpi_root()
        return self.mpicomm.bcast(self[column].min() if isroot else None,root=self.mpiroot)

    @vectorize_columns
    def maximum(self, column):
        if self.is_mpi_scattered():
            return mpi.max_array(self[column],mpicomm=self.mpicomm)
        isroot = self.is_mpi_root()
        return self.mpicomm.bcast(self[column].max() if isroot else None,root=self.mpiroot)

    @vectorize_columns
    def percentile(self, column, q=(15.87,84.13)):
        return self.quantile(column=column,q=np.array(q)/100.)

    @vectorize_columns
    def quantile(self, column, q=(0.1587,0.8413), weights=None):
        if self.is_mpi_scattered():
            return mpi.weighted_quantile_array(self[column],q=q,weights=weights,mpicomm=self.mpicomm)
        isroot = self.is_mpi_root()
        return self.mpicomm.bcast(utils.weighted_quantile(self[column],q=q,weights=weights if isroot else None,root=self.mpiroot))

    def eval(self, literal='None'):
        dglobals = {'np':np,'gsize':self.gsize}
        dglobals.update(self.data)
        return eval(literal,dglobals,{})

    def to_stats(self, columns=None, quantities=None, sigfigs=2, tablefmt='latex_raw', filename=None):
        import tabulate
        #if columns is None: columns = self.columns(exclude='metrics.*')
        if columns is None: columns = self.columns()
        data = []
        if quantities is None: quantities = ['mean','median','std']
        is_latex = 'latex_raw' in tablefmt

        def round_errors(low, up):
            low,up = utils.round_measurement(0.0,low,up,sigfigs=sigfigs)[1:]
            if is_latex: return '${{}}_{{{}}}^{{+{}}}$'.format(low,up)
            return '{}/+{}'.format(low,up)

        for iparam,param in enumerate(parameters):
            row = []
            if is_latex: row.append(param.get_label())
            else: row.append(str(param.name))
            ref_center = self.mean(param)
            ref_error = self.std(param)
            for quantity in quantities:
                if quantity in ['maximum','mean','median','std']:
                    value = getattr(self,quantity)(param)
                    value = utils.round_measurement(value,ref_error,sigfigs=sigfigs)[0]
                    if is_latex: value = '${}$'.format(value)
                    row.append(value)
                else:
                    raise RuntimeError('Unknown quantity {}.'.format(quantity))
            data.append(row)
        tab = tabulate.tabulate(data,headers=quantities,tablefmt=tablefmt)
        if filename and self.is_mpi_root():
            utils.mkdir(os.path.dirname(filename))
            self.log_info('Saving to {}.'.format(filename),rank=self.mpiroot)
            with open(filename,'w') as file:
                file.write(tab)
        return tab
