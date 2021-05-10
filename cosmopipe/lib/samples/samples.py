import os
import re
import glob
import functools
import logging

import numpy as np
import tabulate
from pypescript.mpi import CurrentMPIComm

from cosmopipe.lib.utils import ScatteredBaseClass
from cosmopipe.lib import utils
from cosmopipe.lib.parameter import ParamBlock, Parameter, ParamName

from .utils import *
from cosmopipe.lib import mpi


def vectorize_columns(func):
    @functools.wraps(func)
    def wrapper(self, column, **kwargs):
        if not isinstance(column,list):
            return func(self,column,**kwargs)
        toret = np.asarray([func(self,col,**kwargs) for col in column])
        return toret
    return wrapper


def MPIBroadcast(func):
    @functools.wraps(func)
    @classmethod
    def wrapper(cls, self, *args, mpiroot=0, mpicomm=None, **kwargs):
        new = cls.__new__(cls)
        #if self.mpistate == CurrentMPIState.BROADCAST:
        #    raise MPIError('{} instance already broadcast!'.format(cls.__name__))
        new.mpiroot = mpiroot
        new.mpicomm = mpicomm
        func(new,self,*args,**kwargs)
        new.mpistate = mpi.CurrentMPIState.BROADCAST
        return new
    return wrapper


class Samples(ScatteredBaseClass):

    logger = logging.getLogger('Samples')

    @mpi.MPIInit
    def __init__(self, data=None, parameters=None, attrs=None):
        self.data = {}
        if parameters is None:
            parameters = list((data or {}).keys())
        self.parameters = ParamBlock(parameters)
        if data is not None:
            for name in data:
                self[name] = data[name]
        self.attrs = attrs or {}

    @mpi.MPIBroadcast
    def mpi_broadcast(self, onroot):
        isroot = self.is_mpi_root()
        for key in ['parameters','attrs']:
            setattr(self,key,self.mpicomm.bcast(getattr(onroot,key) if isroot else None,root=self.mpiroot))
        columns = self.mpicomm.bcast(onroot.columns() if isroot else None,root=self.mpiroot)
        self.data = {}
        for col in columns:
            self.data[col] = mpi.broadcast_array(onroot[col] if isroot else None,root=self.mpiroot,mpicomm=self.mpicomm)

    @property
    def size(self):
        return len(self)

    @property
    def gsize(self):
        if self.is_mpi_scattered():
            return self.mpicomm.allreduce(len(self))
        isroot = self.is_mpi_root()
        return self.mpicomm.bcast(len(self) if isroot else None,root=self.mpiroot)

    @mpi.MPIScatter
    def mpi_scatter(self):
        isroot = self.is_mpi_root()
        for key in ['parameters','attrs']:
            setattr(self,key,self.mpicomm.bcast(getattr(self,key) if isroot else None,root=self.mpiroot))
        if not isroot:
            self.data = {}
        columns = self.mpicomm.bcast(self.columns() if isroot else None,root=self.mpiroot)
        for col in columns:
            self[col] = mpi.scatter_array(self[col] if isroot else None,root=self.mpiroot,mpicomm=self.mpicomm)

    @mpi.MPIGather
    def mpi_gather(self):
        for col in self.columns():
            self[col] = mpi.gather_array(self[col],root=self.mpiroot,mpicomm=self.mpicomm)

    def mpi_send(self, dest, tag=42):
        for key in ['mpistate','parameters','attrs']:
            self.mpicomm.send(getattr(self,key),dest=dest,tag=tag)
        self.mpicomm.send(self.columns(),dest=dest,tag=tag)
        for col in self.columns():
            mpi.send_array(self[col],dest=dest,tag=tag,mpicomm=self.mpicomm)

    def mpi_recv(self, source, tag=42):
        for key in ['mpistate','parameters','attrs']:
            setattr(self,key,self.mpicomm.recv(source=source,tag=tag))
        columns = self.mpicomm.recv(source=source,tag=tag)
        for col in columns:
            self[col] = mpi.recv_array(source=source,tag=tag,mpicomm=self.mpicomm)

    @classmethod
    @mpi.CurrentMPIComm.enable
    def mpi_collect(cls, self=None, sources=None, mpicomm=None):
        new = cls.__new__(cls)
        new.mpicomm = mpicomm
        if sources is None:
            issource = self.mpicomm.rank if self is not None else -1
            sources = [rank for rank in new.mpicomm.allgather(issource) if rank >= 0]
        new.mpistate = new.mpicomm.bcast(self.mpistate if new.mpicomm.rank == sources[0] else None,root=sources[0])
        mpiroot = -1
        if (new.mpicomm.rank in sources) and self.is_mpi_root():
            mpiroot = new.mpicomm.rank
        new.mpiroot = [r for r in new.mpicomm.allgather(mpiroot) if r >= 0][0]
        if new.is_mpi_broadcast():
            return cls.mpi_broadcast(self,mpiroot=new.mpiroot,mpicomm=new.mpicomm)
        if new.is_mpi_scattered():
            if new.mpicomm.rank in sources:
                self.mpi_gather()
                self.mpicomm = new.mpicomm
                self.mpiroot = new.mpiroot
                new = self
            new.mpistate = mpi.CurrentMPIState.GATHERED
            new.mpi_scatter()
        return new

    def add_default_parameter(self, name=None):
        param = Parameter(name=name)
        if param.name.tuple[0] == 'metrics':
            param.latex = metrics_to_latex(param.name.tuple[1])
        self.parameters.setdefault(param)

    def columns(self, include=None, exclude=None):
        allcols = list(self.data.keys())

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
            allcols = toret

        return allcols

    def __len__(self):
        return len(self[self.columns()[0]])

    def indices(self):
        sizes = self.mpicomm.allgather(len(self))
        sizes = [0] + np.cumsum(sizes[:1]).tolist()
        return sizes[self.mpicomm.rank] + np.arange(len(self))

    def zeros(self, dtype=np.float64):
        return np.zeros(len(self),dtype=dtype)

    def ones(self, dtype=np.float64):
        return np.ones(len(self),dtype=dtype)

    def falses(self):
        return self.zeros(dtype=np.bool_)

    def trues(self):
        return self.ones(dtype=np.bool_)

    def nans(self):
        return self.ones()*np.nan

    def gslice(self, *args):
        sl = slice(*args)
        isscattered = self.is_mpi_scattered()
        if isscattered:
            self.mpi_gather()
        isroot = self.is_mpi_root() or self.is_mpi_broadcast()
        toret = self
        if isroot:
            toret = self[sl]
        if isscattered:
            toret.mpi_scatter()
        return toret

    def remove_burnin(self, burnin=0):
        if 0 < burnin < 1:
            burnin = burnin*self.gsize
        burnin = int(round(burnin))
        return self.gslice(burnin,None)

    def to_array(self, columns=None):
        if columns is None:
            columns = self.columns()
        return np.array([self[col] for col in columns])

    @classmethod
    def from_array(cls, array, columns, attrs=None, **kwargs):
        return cls({col:arr for col,arr in zip(columns,array)},attrs=attrs,**kwargs)

    def __contains__(self, name):
        return ParamName(name) in self.data

    def __getstate__(self):
        """Return this class state dictionary."""
        data = {str(name):col for name,col in self.data.items()}
        return {'data':data,'parameters':self.parameters.__getstate__(),'attrs':self.attrs}

    def __setstate__(self, state):
        """Set the class state dictionary."""
        self.data = {ParamName(name):col for name,col in state['data'].items()}
        self.parameters = ParamBlock.from_state(state['parameters'])
        self.attrs = state['attrs']

    def __getitem__(self, name):
        if isinstance(name,Parameter):
            return self.data[name.name]
        elif isinstance(name,(ParamName,str,tuple)):
            return self.data[ParamName(name)]
        else:
            return self.__class__({col:self[col][name] for col in self.columns()},parameters=self.parameters,attrs=self.attrs)

    def __setitem__(self, name, item):
        if isinstance(name,Parameter):
            self.data[name.name] = item
            self.parameters.set(name)
        elif isinstance(name,(ParamName,str,tuple)):
            self.data[ParamName(name)] = item
            self.add_default_parameter(name)
        else:
            for col in self.columns():
                self[col][name] = item

    def __repr__(self):
        return 'Samples with length {} and columns {}'.format(len(self),self.columns())

    def extend(self, other):
        new = {}
        if set(other.columns()) != set(self.columns()):
            raise ValueError('Cannot extend samples as columns do not match: {} != {}.'.format(other.columns(),self.columns()))
        for col in self.columns():
            new[col] = np.concatenate([self[col],other[col]],axis=0)
        return new

    def add_default_weight(self, dtype='f8'):
        if ('metrics','weight') not in self:
            self.log_info('Column "weight" not in provided samples. Setting it to 1.',rank=self.mpiroot)
            self['metrics.weight'] = self.ones(dtype=dtype)
        return self['metrics.weight']

    def __eq__(self, other):
        if not isinstance(other,self.__class__):
            return False
        self_columns = self.columns()
        other_columns = other.columns()
        if set(other_columns) != set(self_columns):
            return False
        for self_col,other_col in zip(self_columns,other_columns):
            if np.any(self[self_col] != other[other_col]):
                return False
        return True

    def to_mesh(self, *args, weights=None, **kwargs):
        from .mesh import Mesh
        if self.is_mpi_scattered():
            self.mpi_gather()
        broadcast = self.mpi_broadcast(self)
        if weights is None:
            weights = broadcast.add_default_weight()
        return Mesh.from_samples(broadcast,*args,weights=weights,**kwargs)

    def save_cosmomc(self, base_filename, parameters=None, ichain=None, fmt='%.18e', delimiter=' ', **kwargs):

        isscattered = self.is_mpi_scattered()
        if isscattered: self.mpi_gather()
        if self.is_mpi_root():
            if parameters is None: parameters = self.columns(exclude='metrics.*')
            self.add_default_weight()
            data = self.to_array(columns=['metrics.weight','metrics.logposterior'] + parameters)
            data[1] *= -1
            data = data.T
            utils.mkdir(os.path.dirname(base_filename))
            chain_filename = '{}.txt'.format(base_filename) if ichain is None else '{}_{:d}.txt'.format(base_filename,ichain)
            self.log_info('Saving chain to {}.'.format(chain_filename))
            np.savetxt(chain_filename,data,header='',fmt=fmt,delimiter=delimiter,**kwargs)

            output = ''
            parameters = [self.parameters[param] for param in parameters]
            for param in parameters:
                tmp = '{}* {}\n' if getattr(param,'derived',False) else '{} {}\n'
                output += tmp.format(param.name,param.latex if param.latex is not None else param.name)
            parameters_filename = '{}.paramnames'.format(base_filename)
            self.log_info('Saving parameter names to {}.'.format(parameters_filename))
            with open(parameters_filename,'w') as file:
                file.write(output)

            output = ''
            for param in parameters:
                limits = param.prior.limits
                limits = tuple('N' if limit is None or np.abs(limit) == np.inf else limit for limit in limits)
                output += '{} {} {}\n'.format(param.name,limits[0],limits[1])
            ranges_filename = '{}.ranges'.format(base_filename)
            self.log_info('Saving parameter ranges to {}.'.format(ranges_filename))
            with open(ranges_filename,'w') as file:
                file.write(output)
        if isscattered: self.mpi_scatter()

    @classmethod
    def load_auto(cls, filename, **kwargs):
        if filename.split('.')[-1] == 'txt':
            return cls.load_cosmomc(filename.split('.')[0],**kwargs)
        return cls.load(filename,**kwargs)

    @classmethod
    @mpi.CurrentMPIComm.enable
    def load_cosmomc(cls, base_filename, ichains=None, mpiroot=0, mpistate=mpi.CurrentMPIState.GATHERED, mpicomm=None):

        self = cls(mpiroot=mpiroot,mpistate=mpi.CurrentMPIState.GATHERED,mpicomm=mpicomm)

        if self.is_mpi_root():
            parameters_filename = '{}.paramnames'.format(base_filename)
            self.log_info('Loading parameters file: {}.'.format(parameters_filename))
            self.parameters = ParamBlock()
            with open(parameters_filename) as file:
                for line in file:
                    name,latex = line.split()
                    name = name.strip()
                    if name.endswith('*'): name = name[:-1]
                    latex = latex.strip().replace('\n','')
                    self.parameters.set(Parameter(name=name.strip(),latex=latex,fixed=False))

            ranges_filename = '{}.ranges'.format(base_filename)
            if os.path.exists(ranges_filename):
                self.log_info('Loading parameter ranges from {}.'.format(ranges_filename))
                with open(ranges_filename) as file:
                    for line in file:
                        name,low,high = line.split()
                        latex = latex.replace('\n','')
                        limits = []
                        for lh,li in zip([low,high],[-np.inf,np.inf]):
                            lh = lh.strip()
                            if lh == 'N': lh = li
                            else: lh = float(lh)
                            limits.append(lh)
                        self.parameters[name.strip()].prior.set_limits(limits=limits)
            else:
                self.log_info('Parameter ranges file {} does not exist.'.format(ranges_filename))

            chain_filename = '{}{{}}.txt'.format(base_filename)
            chain_filenames = []
            if ichains is not None:
                if not isinstance(ichains,(tuple,list)):
                    ichains = [ichains]
                for ichain in ichains:
                    chain_filenames.append(chain_filename.format(ichain))
            else:
                chain_filenames = glob.glob(chain_filename.format('*'))

            samples = []
            for chain_filename in chain_filenames:
                self.log_info('Loading chain file: {}.'.format(chain_filename))
                samples.append(np.loadtxt(chain_filename,unpack=True))

            samples = np.concatenate(samples,axis=-1)
            for param,values in zip(self.parameters,samples[2:]):
                  self[param] = values
            self['metrics.weight'] = samples[0]
            self['metrics.logposterior'] = -samples[1]

        self = self.mpi_to_state(mpistate)
        return self

    def to_getdist(self, columns=None):
        from getdist import MCSamples
        isscattered = self.is_mpi_scattered()
        if isscattered: self.mpi_gather()
        toret = None
        if self.is_mpi_root():
            if columns is None:
                columns = self.columns(exclude='metrics.*')
            labels = [self.parameters[col].latex for col in columns]
            weights = self.add_default_weight()
            samples = self.to_array(columns=columns)
            names = [str(col) for col in columns]
            toret = MCSamples(samples=samples.T,weights=weights,loglikes=-self['metrics.logposterior'],names=names,labels=labels)
        if isscattered: self.mpi_scatter()
        return toret

    def cov(self, columns=None, ddof=1, **kwargs):
        if columns is None:
            columns = self.columns()
        weights = self.add_default_weight()
        if self.is_mpi_scattered():
            return mpi.cov_array([self[col] for col in columns],aweights=weights,mpicomm=self.mpicomm)
        isroot = self.is_mpi_root()
        return self.mpicomm.bcast(np.cov([self[col] for col in columns],aweights=weights,ddof=ddof,**kwargs) if isroot else None,root=self.mpiroot)

    def invcov(self, columns=None, **kwargs):
        return utils.inv(self.cov(columns,**kwargs))

    def corrcoef(self, columns=None, **kwargs):
        return utils.cov_to_corrcoef(self.cov(columns,**kwargs))

    def covpair(self, column1, column2, **kwargs):
        return self.cov([column1,column2],**kwargs)[0,1]

    def corrpair(self, column1, column2, **kwargs):
        return self.corrcoef([column1,column2],**kwargs)[0,1]

    @vectorize_columns
    def var(self, column, ddof=1, **kwargs):
        weights = weights = self.add_default_weight()
        if self.is_mpi_scattered():
            return mpi.var_array(self[column],aweights=weights,ddof=ddof,mpicomm=self.mpicomm)
        isroot = self.is_mpi_root()
        return self.mpicomm.bcast(np.sqrt(np.cov(self[column],aweights=weights,ddof=ddof)) if isroot else None,root=self.mpiroot)

    @vectorize_columns
    def std(self, column, **kwargs):
        return np.sqrt(self.var(column,**kwargs))

    def neff(self):
        weights = self.get('metrics.weight',None)
        if weights is None:
            return self.gsize
        if self.is_mpi_scattered():
            s2 = mpi.sum_array(weights,mpicomm=self.mpicomm)**2
            s = mpi.sum_array(weights**2,mpicomm=self.mpicomm)
        else:
            isroot = self.is_mpi_root()
            s2 = self.mpicomm.bcast(np.sum(weights)**2 if isroot else None,root=self.mpiroot)
            s = self.mpicomm.bcast(np.sum(weights**2) if isroot else None,root=self.mpiroot)
        return s2 / s

    @vectorize_columns
    def sum(self, column):
        # NOTE: not weighted!!!
        if self.is_mpi_scattered():
            return mpi.sum_array(self[column],mpicomm=self.mpicomm)
        isroot = self.is_mpi_root()
        return self.mpicomm.bcast(np.sum(self[column],axis=axis) if isroot else None,root=self.mpiroot)

    @vectorize_columns
    def mean(self, column):
        weights = self.add_default_weight()
        if self.is_mpi_scattered():
            return mpi.average_array(self[column],weights=weights,mpicomm=self.mpicomm)
        isroot = self.is_mpi_root()
        return self.mpicomm.bcast(np.average(self[column],weights=weights) if isroot else None,root=self.mpiroot)

    @vectorize_columns
    def median(self, column):
        return self.quantile(column,q=0.5)

    @vectorize_columns
    def minimum(self, column, cost='metrics.chi2'):
        if self.is_mpi_scattered():
            argmin,rank = mpi.argmin_array(self[cost],mpicomm=self.mpicomm)
            return self.mpicomm.bcast(self[column][argmin] if self.mpicomm.rank == rank else None,root=rank)
        isroot = self.is_mpi_root()
        return self.mpicomm.bcast(self[column][np.argmin(self[cost])] if isroot else None,root=self.mpiroot)

    @vectorize_columns
    def maximum(self, column, cost='metrics.logposterior'):
        if self.is_mpi_scattered():
            argmax,rank = mpi.argmax_array(self[cost],mpicomm=self.mpicomm)
            return self.mpicomm.bcast(self[column][argmax] if self.mpicomm.rank == rank else None,root=rank)
        isroot = self.is_mpi_root()
        return self.mpicomm.bcast(self[column][np.argmax(self[cost])] if isroot else None,root=self.mpiroot)

    @vectorize_columns
    def percentile(self, column, q=(15.87,84.13)):
        return self.quantile(column=column,q=np.array(q)/100.)

    def get(self, column, *args):
        if len(args) > 1:
             raise SyntaxError('Too many arguments!')
        if column in self:
            return self[column]
        if args:
            return args[0]
        raise KeyError('Column {} does not exist.'.format(column))

    @vectorize_columns
    def quantile(self, column, q=(0.1587,0.8413)):
        weights = self.add_default_weight()
        if self.is_mpi_scattered():
            return mpi.weighted_quantile_array(self[column],q=q,weights=weights,mpicomm=self.mpicomm)
        isroot = self.is_mpi_root()
        return self.mpicomm.bcast(utils.weighted_quantile(self[column],q=q,weights=weights) if isroot else None,root=self.mpiroot)

    @vectorize_columns
    def interval(self, column, nsigmas=1., bins=100, method='gaussian_kde', bw_method='scott'):
        weights = self.add_default_weight()
        if method == 'exact':
            col = mpi.gather_array(self[column],root=self.mpiroot)
            weights = mpi.gather_array(weights,root=self.mpiroot)
            isroot = self.is_mpi_root()
            if isroot:
                idx = np.argsort(col)
                x = col[idx]
                weights = weights[idx]
                nquantile = nsigmas_to_quantiles_1d(nsigmas)
                cdf = np.cumsum(weights)
                cdf /= cdf[-1]
                cdfpq = cdf+nquantile
                ixmaxlow = np.arange(len(x))
                ixmaxup = np.searchsorted(cdf,cdfpq,side='left')
                mask = ixmaxup < len(x)
                indices = np.array([np.flatnonzero(mask),ixmaxup[mask]])
                xmin,xmax = x[indices]
                argmin = np.argmin(xmax-xmin)
                x = np.array([xmin[argmin],xmax[argmin]])
            return self.mpicomm.bcast(x if isroot else None,root=self.mpiroot)
        isroot = self.is_mpi_root()
        mesh = self.to_mesh(parameters=[column],method=method,bw_method=bw_method,bins=bins)
        if isroot:
            x,pdf = mesh(parameters=[column])
            level = mesh.get_sigmas(nsigmas)[0]
            x = x[pdf>=level]
            x = np.array([x.min(),x.max()])
        return self.mpicomm.bcast(x if isroot else None,root=self.mpiroot)

    def to_stats(self, parameters=None, quantities=None, sigfigs=2, tablefmt='latex_raw', filename=None):
        if parameters is None: parameters = self.columns(exclude='metrics.*')
        parameters = [self.parameters[param] for param in parameters]
        data = []
        if quantities is None: quantities = ['maximum','mean','median','std','quantile:1sigma','interval:1sigma']
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
                elif quantity.startswith('quantile'):
                    nsigmas = int(re.match('quantile:(.*)sigma',quantity).group(1))
                    low,up = self.quantile(param,q=nsigmas_to_quantiles_1d_sym(nsigmas))
                    row.append(round_errors(low-ref_center,up-ref_center))
                elif quantity.startswith('interval'):
                    nsigmas = int(re.match('interval:(.*)sigma',quantity).group(1))
                    low,up = self.interval(param,nsigmas=nsigmas)
                    row.append(round_errors(low-ref_center,up-ref_center))
                else:
                    raise RuntimeError('Unknown quantity {}.'.format(quantity))
            data.append(row)
        tab = tabulate.tabulate(data,headers=quantities,tablefmt=tablefmt)
        if filename and self.is_mpi_root():
            utils.mkdir(os.path.dirname(filename))
            self.log_info('Saving to {}.'.format(filename),rank=0)
            with open(filename,'w') as file:
                file.write(tab)
        return tab
