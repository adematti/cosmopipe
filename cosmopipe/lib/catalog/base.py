"""Definition of :class:`BaseCatalog` to store a catalog-like objects."""

import os
import re
import functools
import logging

import numpy as np

from cosmopipe.lib.utils import ScatteredBaseClass
from cosmopipe.lib import utils, mpi


def vectorize_columns(func):
    """
    Wrapper to vectorize method ``func`` over input columns.
    Uses :meth:`BaseCatalog.vectorize_columns` to check for multiple columns.
    """
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

    """Base class that represents a catalog, as a dictionary of columns stored as arrays."""

    _broadcast_attrs = ['attrs','mpistate','mpiroot']

    @mpi.MPIInit
    def __init__(self, data=None, columns=None, attrs=None):
        """
        Initialize :class:`BaseCatalog`.

        Parameters
        ----------
        data : dict, BaseCatalog
            Dictionary name: array.
            If :class:`BaseCatalog` instance, update ``self`` attributes.

        columns : list, default=None
            List of column names.
            Defaults to ``data.keys()``.

        attrs : dict
            Other attributes.
        """
        if isinstance(data, BaseCatalog):
            self.__dict__.update(data.__dict__)
            return
        self.data = {}
        if columns is None:
            columns = list((data or {}).keys())
        if data is not None:
            for name in columns:
                self[name] = data[name]
        self.attrs = attrs or {}

    @classmethod
    def _multiple_columns(cls, column):
        """Whether ``column`` correspond to multiple columns (list)."""
        return isinstance(column,list)

    @mpi.MPIBroadcast
    def mpi_broadcast(self, onroot):
        """
        Broadcast ``onroot`` catalog on all processes.

        Warning
        -------
        Multiplies global memory footprint by the number of processes.
        """
        isroot = self.is_mpi_root()
        for key in self._broadcast_attrs:
            setattr(self,key,self.mpicomm.bcast(getattr(onroot,key) if isroot else None,root=self.mpiroot))
        columns = self.mpicomm.bcast(list(onroot.data.keys()) if isroot else None,root=self.mpiroot)
        self.data = {}
        for col in columns:
            self[col] = mpi.broadcast_array(onroot[col] if isroot else None,root=self.mpiroot,mpicomm=self.mpicomm)

    @mpi.MPIScatter
    def mpi_scatter(self):
        """Scatter catalog on all processes."""
        isroot = self.is_mpi_root()
        for key in self._broadcast_attrs:
            setattr(self,key,self.mpicomm.bcast(getattr(self,key) if isroot else None,root=self.mpiroot))
        columns = self.columns()
        if not isroot:
            self.data = {}
        for col in columns:
            self[col] = mpi.scatter_array(self[col] if isroot else None,root=self.mpiroot,mpicomm=self.mpicomm)

    def mpi_send(self, dest, tag=42):
        """Send catalog to rank ``dest`` with tag ``tag``."""
        for key in self._broadcast_attrs:
            self.mpicomm.send(getattr(self,key),dest=dest,tag=tag)
        columns = list(self.data.keys())
        self.mpicomm.send(columns,dest=dest,tag=tag)
        for col in columns:
            mpi.send_array(self[col],dest=dest,tag=tag,mpicomm=self.mpicomm)

    def mpi_recv(self, source, tag=42):
        """Receive catalog from rank ``source`` with tag ``tag``."""
        for key in self._broadcast_attrs:
            setattr(self,key,self.mpicomm.recv(source=source,tag=tag))
        columns = self.mpicomm.recv(source=source,tag=tag)
        for col in columns:
            self[col] = mpi.recv_array(source=source,tag=tag,mpicomm=self.mpicomm)

    @mpi.MPIGather
    def mpi_gather(self):
        """
        Gather catalog on a single process.

        Warning
        -------
        May blow up memory of the node this process runs on.
        """
        for col in self.columns():
            self[col] = mpi.gather_array(self[col],root=self.mpiroot,mpicomm=self.mpicomm)

    def __len__(self):
        """Return catalog (local) length (``0`` if no column)."""
        keys = list(self.data.keys())
        if not keys or self[keys[0]] is None:
            return 0
        return len(self[keys[0]])

    @property
    def size(self):
        """Equivalent for :meth:`__length__`."""
        return len(self)

    @property
    def gsize(self):
        """Return catalog global size, i.e. sum of size in each process."""
        if self.is_mpi_scattered():
            return self.mpicomm.allreduce(len(self))
        isroot = self.is_mpi_root()
        return self.mpicomm.bcast(len(self) if isroot else None,root=self.mpiroot)

    def columns(self, include=None, exclude=None):
        """
        Return catalog column names, after optional selections.

        Parameters
        ----------
        include : list, string, default=None
            Single or list of *regex* patterns to select column names to include.
            Defaults to all columns.

        exclude : list, string, default=None
            Single or list of *regex* patterns to select column names to exclude.
            Defaults to no columns.

        Returns
        -------
        columns : list
            Return catalog column names, after optional selections.
        """
        toret = None

        if self.is_mpi_root():
            toret = allcols = [str(name) for name in self.data]

            if include is not None:
                if not isinstance(include,(tuple,list)):
                    include = [include]
                toret = []
                for inc in include:
                    for col in allcols:
                        if re.match(inc,str(col)):
                            toret.append(col)
                allcols = toret

            if exclude is not None:
                if not isinstance(exclude,(tuple,list)):
                    exclude = [exclude]
                toret = []
                for exc in exclude:
                    for col in allcols:
                        if re.match(exc,str(col)) is None:
                            toret.append(col)

        return self.mpicomm.bcast(toret,root=self.mpiroot)

    def __contains__(self, column):
        """Whether catalog contains column name ``column``."""
        return column in self.data

    def gindices(self):
        """Row numbers in the global catalog."""
        sizes = self.mpicomm.allgather(len(self))
        sizes = [0] + np.cumsum(sizes[:1]).tolist()
        return sizes[self.mpicomm.rank] + np.arange(len(self))

    def zeros(self, dtype=np.float64):
        """Return array of size :attr:`size` filled with zero."""
        return np.zeros(len(self),dtype=dtype)

    def ones(self, dtype=np.float64):
        """Return array of size :attr:`size` filled with one."""
        return np.ones(len(self),dtype=dtype)

    def full(self, fill_value, dtype=np.float64):
        """Return array of size :attr:`size` filled with ``fill_value``."""
        return np.full(len(self),fill_value,dtype=dtype)

    def falses(self):
        """Return array of size :attr:`size` filled with ``False``."""
        return self.zeros(dtype=np.bool_)

    def trues(self):
        """Return array of size :attr:`size` filled with ``True``."""
        return self.ones(dtype=np.bool_)

    def nans(self):
        """Return array of size :attr:`size` filled with :attr:`numpy.nan`."""
        return self.ones()*np.nan

    def get(self, column, *args, **kwargs):
        """Return catalog (local) column ``column`` if exists, else return provided default."""
        has_default = False
        if args:
            if len(args) > 1:
                raise SyntaxError('Too many arguments')
            has_default = True
            default = args[0]
        if kwargs:
            if len(kwargs) > 1:
                raise SyntaxError('Too many arguments')
            has_default = True
            default = kwargs['default']
        if column not in self.data and has_default:
            return default
        return self.data[column]

    def set(self, column, item):
        """Set column of name ``column``."""
        self.data[column] = item

    def gget(self, column, root=None):
        """
        Return on process rank ``root`` catalog global column ``column`` if exists, else return provided default.
        If ``root`` is ``None`` or ``Ellipsis`` return result on all processes.
        """
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
        """
        Perform global slicing of catalog,
        e.g. ``catalog.gslice(0,100,1)`` will return a new catalog of global size ``100``.
        Same reference to :attr:`attrs`.
        """
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
        """
        Return catalog as *numpy* array.

        Parameters
        ----------
        columns : list, default=None
            Columns to use. Defaults to all catalog columns.

        struct : bool, default=True
            Whether to return structured array, with columns accessible through e.g. ``array['Position']``.
            If ``False``, *numpy* will attempt to cast types of different columns.

        Returns
        -------
        array : array
        """
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
        """
        Build :class:`BaseCatalog` from input ``array``.

        Parameters
        ----------
        columns : list
            List of columns to read from array.

        mpiroot : int, default=0
            Rank of process where input array lives.

        mpistate : string, mpi.CurrentMPIState
            MPI state of the input array: 'scattered', 'gathered', 'broadcast'?

        mpicomm : MPI communicator, default=None
            MPI communicator.

        kwargs : dict
            Other arguments for :meth:`__init__`.

        Returns
        -------
        catalog : BaseCatalog
        """
        isstruct = None
        if mpicomm.rank == mpiroot:
            isstruct = array.dtype.names is not None
            if isstruct:
                if columns is None: columns = array.dtype.names
        isstruct = mpicomm.bcast(isstruct,root=mpiroot)
        columns = mpicomm.bcast(columns,root=mpiroot)
        if columns is None:
            raise ValueError('Could not find columns in provided array. Please provide columns.')
        new = cls(data=dict.fromkeys(columns),mpiroot=mpiroot,mpistate=mpistate,mpicomm=mpicomm,**kwargs)
        if new.is_mpi_gathered() and not new.is_mpi_root():
            return new
        if isstruct:
            new.data = {col:array[col] for col in columns}
        else:
            new.data = {col:arr for col,arr in zip(columns,array)}
        return new

    def copy(self, columns=None):
        """Return copy, including column names ``columns`` (defaults to all columns)."""
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
        """Get catalog column ``name`` if string, else return copy with local slice."""
        if isinstance(name,str):
            return self.get(name)
        new = self.copy()
        new.data = {col:self[col][name] for col in self.data}
        return new

    def __setitem__(self, name, item):
        """Set catalog column ``name`` if string, else set slice ``name`` of all columns to ``item``."""
        if isinstance(name,str):
            return self.set(name,item)
        for col in self.data:
            self[col][name] = item

    def __delitem__(self, name):
        """Delete column ``name``."""
        del self.data[name]

    def __repr__(self):
        """Return string representation of catalog, including global size and columns."""
        return '{}(size={:d}, columns={})'.format(self.__class__.__name__,self.gsize,self.columns())

    @classmethod
    def concatenate(cls, *others):
        """
        Concatenate catalogs together.

        Parameters
        ----------
        others : list
            List of :class:`BaseCatalog` instances.

        Returns
        -------
        new : BaseCatalog

        Warning
        -------
        :attr:`attrs` of returned catalog contains, for each key, the last value found in ``others`` :attr:`attrs` dictionaries.
        """
        attrs = {}
        for other in others: attrs.update(other.attrs)
        others = [other for other in others if other.columns()]

        new = others[0].copy()
        new.attrs = attrs
        new_columns = new.columns()

        for other in others:
            other_columns = other.columns()
            assert new.mpicomm is other.mpicomm
            if new_columns and other_columns and set(other_columns) != set(new_columns):
                raise ValueError('Cannot extend samples as columns do not match: {} != {}.'.format(other_columns,new_columns))

        for column in new_columns:
            columns = [other.gget(column,root=new.mpiroot) for other in others]
            if new.is_mpi_root():
                new[column] = np.concatenate(columns,axis=0)
            if new.is_mpi_scattered():
                new[column] = mpi.scatter_array(new[column] if new.is_mpi_root() else None,root=new.mpiroot,mpicomm=new.mpicomm)
            elif new.is_mpi_broadcast():
                new[column] = mpi.broadcast_array(new[column] if new.is_mpi_root() else None,root=new.mpiroot,mpicomm=new.mpicomm)
        return new

    def extend(self, other):
        """Extend catalog with ``other``."""
        new = self.concatenate(self,other)
        self.__dict__.update(new.__dict__)

    def __eq__(self, other):
        """Is ``self`` equal to ``other``, i.e. same type and columns? (ignoring :attr:`attrs`)"""
        if not type(other) == type(self):
            return False
        #if other.attrs != self.attrs:
        #    return False
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

    @classmethod
    def load_auto(cls, filename, *args, **kwargs):
        """
        Load catalog from disk.

        Parameters
        ----------
        filename : string
            File name of catalog.
            If ends with '.fits', calls :meth:`load_fits`.
            Else (*numpy* binary format), calls :meth:`load`.

        args : list
            Arguments for load function.

        kwargs : dict
            Other arguments for load function.
        """
        if os.path.splitext(filename)[-1] == '.fits':
            return cls.load_fits(filename,*args,**kwargs)
        return cls.load(filename,**kwargs)

    def save_auto(self, filename, *args, **kwargs):
        """
        Write catalog to disk.

        Parameters
        ----------
        filename : string
            File name of catalog.
            If ends with '.fits', calls :meth:`save_fits`.
            Else (*numpy* binary format), calls :meth:`save`.

        args : list
            Arguments for save function.

        kwargs : dict
            Other arguments for save function.
        """
        if os.path.splitext(filename)[-1] == '.fits':
            return self.save_fits(filename,*args,**kwargs)
        return self.save(filename)

    @classmethod
    def load(cls, *args, **kwargs):
        """Load catalog in *numpy* binary format from disk."""
        self = super(BaseCatalog,cls).load(*args,**kwargs)
        for key in self._broadcast_attrs:
            setattr(self,key,self.mpicomm.bcast(getattr(self,key) if self.is_mpi_root() else None,root=self.mpiroot))
        for col in self.columns():
            if col not in self: self.data[col] = None
        return self

    @classmethod
    @mpi.CurrentMPIComm.enable
    def load_fits(cls, filename,  columns=None, ext=None, mpiroot=0, mpistate=mpi.CurrentMPIState.SCATTERED, mpicomm=None):
        """
        Load catalog in *fits* binary format from disk.

        Parameters
        ----------
        columns : list, default=None
            List of column names to read. Defaults to all columns.

        ext : int, default=None
            *fits* extension. Defaults to first extension with data.

        mpiroot : int, default=0
            Rank of process where input array lives.

        mpistate : string, mpi.CurrentMPIState
            MPI state of the input array: 'scattered', 'gathered', 'broadcast'?

        mpicomm : MPI communicator, default=None
            MPI communicator.

        Returns
        -------
        catalog : BaseCatalog
        """
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
        """Save catalog to ``filename`` as *fits* file. Possible to change fitsio to write by chunks?."""
        import fitsio
        array = self.to_array(struct=True)
        if self.is_mpi_scattered():
            array = mpi.gather_array(array,root=self.mpiroot,mpicomm=self.mpicomm)
        if self.is_mpi_root():
            fitsio.write(filename,array,header=self.attrs.get('fitshdr',None),clobber=True)

    @classmethod
    def from_nbodykit(cls, catalog, columns=None):
        """
        Build new catalog from **nbodykit**.

        Parameters
        ----------
        catalog : nbodykit.base.catalog.CatalogSource
            **nbodykit** catalog.

        columns : list, default=None
            Columns to import. Defaults to all columns.

        Returns
        -------
        catalog : BaseCatalog
        """
        if columns is None: columns = catalog.columns
        data = {col: catalog[col].compute() for col in columns}
        return cls(data,mpicomm=catalog.comm,mpistate='scattered',mpiroot=0,attrs=catalog.attrs)

    def to_nbodykit(self, columns=None):
        """
        Return catalog in **nbodykit** format.

        Parameters
        ----------
        columns : list, default=None
            Columns to export. Defaults to all columns.

        Returns
        -------
        catalog : nbodykit.base.catalog.CatalogSource
        """
        if columns is None: columns = self.columns()
        if not self.is_mpi_scattered():
            self = self.copy()
            self.mpi_scatter()
        source = {col:self[col] for col in columns}
        from nbodykit.lab import ArrayCatalog
        return ArrayCatalog(source,**self.attrs)

    @vectorize_columns
    def sum(self, column):
        """Return global sum of column(s) ``column``."""
        # NOTE: not weighted!!!
        if self.is_mpi_scattered():
            return mpi.sum_array(self[column],mpicomm=self.mpicomm)
        isroot = self.is_mpi_root()
        return self.mpicomm.bcast(np.sum(self[column]) if isroot else None,root=self.mpiroot)

    def cov(self, columns=None, fweights=None, aweights=None, ddof=1):
        """
        Estimate weighted covariance.

        Parameters
        ----------
        columns : list, default=None
            Columns to compute covariance for.

        fweights : array, int, default=None
            1D array of integer frequency weights; the number of times each
            observation vector should be repeated.

        aweights : array, default=None
            1D array of observation vector weights. These relative weights are
            typically large for observations considered "important" and smaller for
            observations considered less "important". If ``ddof=0`` the array of
            weights can be used to assign probabilities to observation vectors.

        ddof : int, default=1
            Number of degrees of freedom.

        Returns
        -------
        cov : scalar, array
            If single parameter provided as ``columns``, returns variance for that parameter (scalar).
            Else returns covariance (2D array).
        """
        if columns is None: columns = self.columns(varied=True)
        isscalar = not self._multiple_columns(columns)
        if isscalar: columns = [columns]
        if self.is_mpi_scattered():
            toret = mpi.cov_array([self[col] for col in columns],fweights=fweights,aweights=aweights,mpicomm=self.mpicomm)
            #if isscalar: return toret[0]
            if len(columns) == 1 and not isscalar:
                toret = np.atleast_2d(toret)
            return toret
        isroot = self.is_mpi_root()
        toret = self.mpicomm.bcast(np.cov([self[col] for col in columns],fweights=fweights,aweights=aweights,ddof=ddof) if isroot else None,root=self.mpiroot)
        #if isscalar: return toret[0]
        if len(columns) == 1 and not isscalar:
            toret = np.atleast_2d(toret)
        return toret

    @vectorize_columns
    def var(self, column, fweights=None, aweights=None, ddof=1):
        """
        Estimate weighted parameter variance.

        Parameters
        ----------
        columns : list, default=None
            Columns to compute variance for.

        fweights : array, int, default=None
            1D array of integer frequency weights; the number of times each
            observation vector should be repeated.

        aweights : array, default=None
            1D array of observation vector weights. These relative weights are
            typically large for observations considered "important" and smaller for
            observations considered less "important". If ``ddof=0`` the array of
            weights can be used to assign probabilities to observation vectors.

        Returns
        -------
        var : scalar, array
            If single parameter provided as ``columns``, returns variance for that parameter (scalar).
            Else returns variance array.
        """
        if self.is_mpi_scattered():
            return mpi.var_array(self[column],axis=0,fweights=fweights,aweights=aweights,ddof=ddof,mpicomm=self.mpicomm)
        isroot = self.is_mpi_root()
        return self.mpicomm.bcast(np.cov(self[column],fweights=fweights,aweights=aweights,ddof=ddof) if isroot else None,root=self.mpiroot)

    @vectorize_columns
    def std(self, column, **kwargs):
        """
        Estimate weigthed standard deviation.
        Same arguments as :meth:`var`.
        """
        return np.sqrt(self.var(column,**kwargs))

    @vectorize_columns
    def average(self, column, weights=None):
        """Return global average of column(s) ``column``, with weights ``weights`` (defaults to ``1``)."""
        if self.is_mpi_scattered():
            return mpi.average_array(self[column],weights=weights,mpicomm=self.mpicomm)
        isroot = self.is_mpi_root()
        return self.mpicomm.bcast(np.average(self[column],weights=weights) if isroot else None,root=self.mpiroot)

    @vectorize_columns
    def mean(self, column):
        """Return global mean of column(s) ``column``."""
        return self.average(column)

    @vectorize_columns
    def median(self, column):
        """Return global median of column(s) ``column``."""
        return self.quantile(column,q=0.5)

    @vectorize_columns
    def minimum(self, column):
        """Return global minimum of column(s) ``column``."""
        if self.is_mpi_scattered():
            return mpi.min_array(self[column],mpicomm=self.mpicomm)
        isroot = self.is_mpi_root()
        return self.mpicomm.bcast(self[column].min() if isroot else None,root=self.mpiroot)

    @vectorize_columns
    def maximum(self, column):
        """Return global maximum of column(s) ``column``."""
        if self.is_mpi_scattered():
            return mpi.max_array(self[column],mpicomm=self.mpicomm)
        isroot = self.is_mpi_root()
        return self.mpicomm.bcast(self[column].max() if isroot else None,root=self.mpiroot)

    @vectorize_columns
    def quantile(self, column, q=(0.1587,0.8413), weights=None):
        """Return global quantiles of column(s) ``column``."""
        if self.is_mpi_scattered():
            return mpi.weighted_quantile_array(self[column],q=q,weights=weights,mpicomm=self.mpicomm)
        isroot = self.is_mpi_root()
        return self.mpicomm.bcast(utils.weighted_quantile(self[column],q=q,weights=weights) if isroot else None,root=self.mpiroot)

    @vectorize_columns
    def percentile(self, column, q=(15.87,84.13)):
        """Return global percentiles of column(s) ``column``."""
        return self.quantile(column=column,q=np.array(q)/100.)

    def eval(self, literal='None'):
        """
        Evaluate input ``literal`` and return results.
        Python's :func:`eval` is provided access to *numpy* (``np``),
        catalog global size :attr:`gsize` and columns.
        """
        dglobals = {'np':np,'gsize':self.gsize}
        dglobals.update(self.data)
        return eval(literal,dglobals,{})

    def to_stats(self, columns=None, quantities=None, sigfigs=2, tablefmt='latex_raw', filename=None):
        """
        Export catalog summary quantities.

        Parameters
        ----------
        columns : list, default=None
            Columns to export quantities for.
            Defaults to all columns.

        quantities : list, default=None
            Quantities to export. Defaults to ``['mean','median','std']``.

        sigfigs : int, default=2
            Number of significant digits.
            See :func:`utils.round_measurement`.

        tablefmt : string, default='latex_raw'
            Format for summary table.
            See :func:`tabulate.tabulate`.

        filename : string default=None
            If not ``None``, file name where to save summary table.

        Returns
        -------
        tab : string
            Summary table.
        """
        import tabulate
        #if columns is None: columns = self.columns(exclude='metrics.*')
        if columns is None: columns = self.columns()
        data = []
        if quantities is None: quantities = ['mean','median','std']
        is_latex = 'latex_raw' in tablefmt

        for column in columns:
            row = []
            row.append(column)
            ref_center = self.mean(column)
            ref_error = self.std(column)
            for quantity in quantities:
                if quantity in ['maximum','mean','median','std']:
                    value = getattr(self,quantity)(column)
                    value = utils.round_measurement(value,ref_error,sigfigs=sigfigs)[0]
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
