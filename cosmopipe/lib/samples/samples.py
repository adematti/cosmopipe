import os
import re
import glob
import functools
import logging

import numpy as np

from cosmopipe.lib import utils, mpi
from cosmopipe.lib.catalog import BaseCatalog
from cosmopipe.lib.parameter import ParamBlock, Parameter, ParamName

from .utils import *


def _multiple_columns(column):
    return isinstance(column,(list,ParamBlock))


def vectorize_columns(func):
    @functools.wraps(func)
    def wrapper(self, column, **kwargs):
        if not _multiple_columns(column):
            return func(self,column,**kwargs)
        toret = [func(self,col,**kwargs) for col in column]
        if all(t is None for t in toret): # in case not broadcast to all ranks
            return None
        return np.asarray(toret)
    return wrapper


class hybridmethod:
    """Taken from https://stackoverflow.com/questions/28237955/same-name-for-classmethod-and-instancemethod."""

    def __init__(self, fclass, finstance=None, doc=None):
        self.fclass = fclass
        self.finstance = finstance
        self.__doc__ = doc or fclass.__doc__
        # support use on abstract base classes
        self.__isabstractmethod__ = bool(getattr(fclass, '__isabstractmethod__', False))

    def classmethod(self, fclass):
        return type(self)(fclass, self.finstance, None)

    def instancemethod(self, finstance):
        return type(self)(self.fclass, finstance, self.__doc__)

    def __get__(self, instance, cls):
        if instance is None or self.finstance is None:
              # either bound to the class, or no instance method available
            return self.fclass.__get__(cls, None)
        return self.finstance.__get__(instance, cls)


class Samples(BaseCatalog):

    logger = logging.getLogger('Samples')
    _broadcast_attrs = ['parameters','attrs','mpistate','mpiroot']

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

    def add_default_parameter(self, name=None):
        param = Parameter(name=name)
        if param.name.tuple[0] == 'metrics':
            param.latex = metrics_to_latex(param.name.tuple[1])
        self.parameters.setdefault(param)

    def columns(self, include=None, exclude=None, **kwargs):
        toret = super(Samples,self).columns(include=include,exclude=exclude)

        if kwargs and self.is_mpi_root():
            parameter_selection = self.parameters.select(**kwargs)
            toret = [col for col in toret if col in parameter_selection]

        return self.mpicomm.bcast(toret,root=self.mpiroot)

    def get(self, name, *args, **kwargs):
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
        if isinstance(name,Parameter):
            paramname = name.name
        else:
            paramname = ParamName(name)
        if paramname not in self.data:
            for name in [('metrics','fweight'),('metrics','aweight')]:
                if name == paramname.tuple:
                    return self.ones(dtype='f8')
            if paramname.tuple == ('metrics','weight'):
                return self['metrics.fweight']*self['metrics.aweight']
            if has_default:
                return default
        return self.data[paramname]

    def set(self, name, item):
        if isinstance(name,Parameter):
            self.data[name.name] = item
            self.parameters.set(name)
        else:
            self.data[ParamName(name)] = item
            self.add_default_parameter(name)

    def remove_burnin(self, burnin=0):
        if 0 < burnin < 1:
            burnin = burnin*self.gsize
        burnin = int(round(burnin))
        return self.gslice(burnin,None)

    def __contains__(self, name):
        return ParamName(name) in self.data

    def __getstate__(self):
        """Return this class state dictionary."""
        data = {str(name):col for name,col in self.data.items()}
        return {'data':data,'parameters':self.parameters.__getstate__(),'attrs':self.attrs}

    def __setstate__(self, state):
        """Set the class state dictionary."""
        self.data = state['data'].copy()
        self.parameters = ParamBlock.from_state(state['parameters'])
        self.attrs = state['attrs']

    def __getitem__(self, name):
        if isinstance(name,(Parameter,ParamName,str,tuple)):
            return self.get(name)
        new = self.copy()
        new.data = {col:self[col][name] for col in self.data}
        return new

    def __setitem__(self, name, item):
        if isinstance(name,(Parameter,ParamName,str,tuple)):
            return self.set(name,item)
        for col in self.data:
            self[col][name] = item

    def __repr__(self):
        return 'Samples(size={:d}, columns={})'.format(self.gsize,self.columns())

    def to_array(self, columns=None, struct=True):
        if self.is_mpi_gathered() and not self.is_mpi_root():
            return None
        if columns is None:
            columns = self.columns()
        columns = [col.name if isinstance(col,Parameter) else ParamName(col) for col in columns]
        if struct:
            toret = np.empty(self.size,dtype=[(str(col),self[col].dtype,self[col].shape[1:]) for col in columns])
            for col in columns: toret[str(col)] = self[col]
            return toret
        return np.array([self[col] for col in columns])

    def to_mesh(self, columns, **kwargs):
        from .mesh import Mesh
        if columns is None: columns = self.columns(fixed=False)
        if not _multiple_columns(columns):
            columns = [columns]
        columns = list(columns) + ['metrics.weight']
        samples = []
        for col in columns: samples.append(self.gget(col))
        return Mesh.from_samples(samples[:-1],weights=samples[-1],names=columns,**kwargs)

    @classmethod
    def load_auto(cls, filename, *args, **kwargs):
        if os.path.splitext(filename)[-1] == '.txt':
            return cls.load_cosmomc(os.path.splitext(filename)[0],*args,**kwargs)
        return cls.load(filename,*args,**kwargs)

    def save_auto(self, filename, *args, **kwargs):
        if os.path.splitext(filename)[-1] == '.txt':
            return self.save_cosmomc(os.path.splitext(filename)[0],*args,**kwargs)
        return self.save(filename)

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
            self['metrics.aweight'] = samples[0]
            self['metrics.logposterior'] = -samples[1]
        for key in cls._broadcast_attrs:
            setattr(self,key,self.mpicomm.bcast(getattr(self,key) if self.is_mpi_root() else None,root=self.mpiroot))
        for col in self.columns():
            if col not in self: self.data[col] = None
        self = self.mpi_to_state(mpistate)
        return self

    def save_cosmomc(self, base_filename, columns=None, ichain=None, fmt='%.18e', delimiter=' ', **kwargs):

        isscattered = self.is_mpi_scattered()
        if isscattered: self.mpi_gather()
        if columns is None: columns = self.columns(exclude='metrics.*')
        if self.is_mpi_root():
            data = self.to_array(columns=['metrics.weight','metrics.logposterior'] + columns,struct=False)
            data[1] *= -1
            data = data.T
            utils.mkdir(os.path.dirname(base_filename))
            chain_filename = '{}.txt'.format(base_filename) if ichain is None else '{}_{:d}.txt'.format(base_filename,ichain)
            self.log_info('Saving chain to {}.'.format(chain_filename))
            np.savetxt(chain_filename,data,header='',fmt=fmt,delimiter=delimiter,**kwargs)

            output = ''
            parameters = [self.parameters[col] for col in columns]
            for param in parameters:
                tmp = '{}* {}\n' if getattr(param,'derived',getattr(param,'fixed')) else '{} {}\n'
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
        self.mpicomm.Barrier()
        if isscattered: self.mpi_scatter()

    def to_getdist(self, columns=None):
        from getdist import MCSamples
        isscattered = self.is_mpi_scattered()
        if isscattered: self.mpi_gather()
        toret = None
        if columns is None: columns = self.columns(exclude='metrics.*')
        if self.is_mpi_root():
            labels = [self.parameters[col].latex for col in columns]
            samples = self.to_array(columns=columns,struct=False)
            names = [str(col) for col in columns]
            toret = MCSamples(samples=samples.T,weights=self['metrics.weight'],loglikes=-self['metrics.logposterior'],names=names,labels=labels)
        if isscattered: self.mpi_scatter()
        return toret

    def cov(self, columns=None, ddof=1, **kwargs):
        if columns is None: columns = self.columns(fixed=False)
        isscalar = not _multiple_columns(columns)
        if isscalar: columns = [columns]
        if self.is_mpi_scattered():
            toret = mpi.cov_array([self[col] for col in columns],fweights=self['metrics.fweight'],aweights=self['metrics.aweight'],mpicomm=self.mpicomm)
            #if isscalar: return toret[0]
            if len(columns) == 1 and not isscalar:
                toret = np.atleast_2d(toret)
            return toret
        isroot = self.is_mpi_root()
        toret = self.mpicomm.bcast(np.cov([self[col] for col in columns],fweights=self['metrics.fweight'],aweights=self['metrics.aweight'],ddof=ddof,**kwargs) if isroot else None,root=self.mpiroot)
        #if isscalar: return toret[0]
        if len(columns) == 1 and not isscalar:
            toret = np.atleast_2d(toret)
        return toret

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
        if self.is_mpi_scattered():
            return mpi.var_array(self[column],fweights=self['metrics.fweight'],aweights=self['metrics.aweight'],ddof=ddof,mpicomm=self.mpicomm)
        isroot = self.is_mpi_root()
        return self.mpicomm.bcast(np.sqrt(np.cov(self[column],fweights=self['metrics.fweight'],aweights=self['metrics.aweight'],ddof=ddof)) if isroot else None,root=self.mpiroot)

    @vectorize_columns
    def std(self, column, **kwargs):
        return np.sqrt(self.var(column,**kwargs))

    @vectorize_columns
    def sum(self, column):
        # NOTE: not weighted!!!
        if self.is_mpi_scattered():
            return mpi.sum_array(self[column],mpicomm=self.mpicomm)
        isroot = self.is_mpi_root()
        return self.mpicomm.bcast(np.sum(self[column]) if isroot else None,root=self.mpiroot)

    @vectorize_columns
    def mean(self, column):
        if self.is_mpi_scattered():
            return mpi.average_array(self[column],weights=self['metrics.weight'],mpicomm=self.mpicomm)
        isroot = self.is_mpi_root()
        return self.mpicomm.bcast(np.average(self[column],weights=self['metrics.weight']) if isroot else None,root=self.mpiroot)

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

    @vectorize_columns
    def quantile(self, column, q=(0.1587,0.8413)):
        if self.is_mpi_scattered():
            return mpi.weighted_quantile_array(self[column],q=q,weights=self['metrics.weight'],mpicomm=self.mpicomm)
        isroot = self.is_mpi_root()
        return self.mpicomm.bcast(utils.weighted_quantile(self[column],q=q,weights=self['metrics.weight']) if isroot else None,root=self.mpiroot)

    @vectorize_columns
    def interval(self, column, nsigmas=1., bins=100, method='gaussian_kde', bw_method='scott'):
        isroot = self.is_mpi_root()

        if method == 'exact':
            if self.is_mpi_scattered():
                col = mpi.gather_array(self[column],root=self.mpiroot)
                weights = mpi.gather_array(self['metrics.weight'],root=self.mpiroot)
            elif isroot:
                col = self[column]
                weights = self['metrics.weight']
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

        mesh = self.to_mesh([column],method=method,bw_method=bw_method,bins=bins)

        if isroot:
            x,pdf = mesh([column])
            level = mesh.get_sigmas(nsigmas)[0]
            x = x[pdf>=level]
            x = np.array([x.min(),x.max()])

        return self.mpicomm.bcast(x if isroot else None,root=self.mpiroot)

    def to_stats(self, columns=None, quantities=None, sigfigs=2, tablefmt='latex_raw', filename=None):
        import tabulate
        #if columns is None: columns = self.columns(exclude='metrics.*')
        if columns is None: columns = self.columns(fixed=False)
        parameters = [self.parameters[col] for col in columns]
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
            self.log_info('Saving to {}.'.format(filename),rank=self.mpiroot)
            with open(filename,'w') as file:
                file.write(tab)
        return tab

    @classmethod
    def gelman_rubin(cls, chains, columns=None, statistic='mean', method='eigen', return_matrices=False, check=True):
        """
        http://www.stat.columbia.edu/~gelman/research/published/brooksgelman2.pdf
        """
        if columns is None: columns = self.columns(fixed=False)
        isscalar = not _multiple_columns(columns)
        if isscalar: columns = [columns]

        if not isinstance(chains,(list,tuple)):
            raise ValueError('Provide a list of at least 2 chains to estimate Gelman-Rubin'.format(nchains))
        nchains = len(chains)
        if nchains < 2:
            raise ValueError('{:d} chains provided; one needs at least 2 to estimate Gelman-Rubin'.format(nchains))

        if statistic == 'mean':
            def statistic(samples, columns):
                return samples.mean(columns)

        means = np.asarray([statistic(samples,columns) for samples in chains])
        covs = np.asarray([samples.cov(columns) for samples in chains])
        nsteps = np.asarray([samples.sum('metrics.weight') if 'metrics.weight' in samples else samples.gsize for samples in chains])
        # W = "within"
        Wn1 = np.average(covs,weights=nsteps,axis=0)
        Wn = np.average(((nsteps-1.)/nsteps)[:,None,None]*covs,weights=nsteps,axis=0)
        # B = "between"
        # We don't weight with the number of samples in the chains here:
        # shorter chains will likely be outliers, and we want to notice them
        B = np.cov(means.T,ddof=1)
        V = Wn + (nchains + 1.)/nchains*B
        if method == 'eigen':
            # divide by stddev for numerical stability
            stddev = np.sqrt(np.diag(V).real)
            V = V/stddev[:,None]/stddev[None,:]
            invWn1 = utils.inv(Wn1/stddev[:,None]/stddev[None,:],check=check)
            toret = np.linalg.eigvalsh(invWn1.dot(V))
        else:
            toret = np.diag(V)/np.diag(Wn1)
        if isscalar:
            toret = toret[0]
        if return_matrices:
            return toret, (V, Wn1)
        return toret

    @vectorize_columns
    def autocorrelation(self, column, weight_units=True):
        isroot = self.is_mpi_root()
        if self.is_mpi_scattered():
            col = mpi.gather_array(self[column],root=self.mpiroot,mpicomm=self.mpicomm)
            weights = mpi.gather_array(self['metrics.weight'],root=self.mpiroot,mpicomm=self.mpicomm)
        elif isroot:
            col = self[column]
            weights = self['metrics.weight']
        autocorr = None
        if isroot:
            x = (col - np.average(col,weights=weights))*weights
            autocorr = _autocorrelation_1d(x)
            if not weight_units:
                autocorr *= weights.sum()/weights.size
        if self.is_mpi_scattered():
            return mpi.scatter_array(autocorr,root=self.mpiroot,mpicomm=self.mpicomm)
        if self.is_mpi_broadcast():
            return mpi.broadcast_array(autocorr,root=self.mpiroot,mpicomm=self.mpicomm)
        return autocorr

    @hybridmethod
    def integrated_autocorrelation_time(cls, chains, column, min_corr=None, c=5, reliable=50, check=False):
        """
        Taken from: https://github.com/dfm/emcee/blob/master/emcee/autocorr.py
        """
        if not isinstance(chains,(list,tuple)):
            chains = [chains]

        if _multiple_columns(column):
            return np.array([cls.integrated_autocorrelation_time(chains,col,min_corr=min_corr,c=c,reliable=reliable,check=check) for col in column])

        # Automated windowing procedure following Sokal (1989)
        def auto_window(taus, c):
            m = np.arange(len(taus)) < c * taus
            if np.any(m):
                return np.argmin(m)
            return len(taus) - 1

        mpiroot = chains[0].mpiroot
        isroot = chains[0].is_mpi_root()
        mpicomm = chains[0].mpicomm
        gsize = chains[0].gsize

        corr = 0
        for samples in chains:
            if samples.gsize != gsize:
                raise ValueError('Input chains must have same length')
            corr_ = samples.autocorrelation(column,weight_units=True)
            if samples.is_mpi_scattered():
                corr_ = mpi.gather_array(corr_,root=mpiroot,mpicomm=mpicomm)
            elif samples.is_mpi_gathered() and samples.mpiroot != mpiroot:
                if samples.is_mpi_root(): mpi.send_array(corr_,dest=mpiroot,tag=42,mpicomm=mpicomm)
                if isroot: corr_ = mpi.recv_array(source=samples.mpiroot,tag=42,mpicomm=mpicomm)
            if isroot:
                corr += corr_
        if isroot:
            corr = corr/len(chains)
        toret = None
        if min_corr is not None:
            if isroot:
                ix = np.argmin(corr > min_corr * corr[0])
                toret = 1 + 2 * np.sum(corr[1:ix])
        elif c is not None:
            if isroot:
                taus = 2 * np.cumsum(corr) - 1 # 1 + 2 sum_{i=1}^{N} f_{i}
                window = auto_window(taus, c)
                toret = taus[window]
        else:
            raise ValueError('A criterion must be provided to stop integration of correlation time')
        toret = mpicomm.bcast(toret,root=mpiroot)
        if check and reliable * toret > gsize:
            msg = 'The chain is shorter than {:d} times the integrated autocorrelation time for {}. Use this estimate with caution and run a longer chain!\n'.format(reliable,column)
            msg += 'N/{:d} = {:.0f};\ntau: {}'.format(reliable,gsize*1./reliable,toret)
            cls.log_warning(msg,rank=mpiroot)
        return toret

    @integrated_autocorrelation_time.instancemethod
    def integrated_autocorrelation_time(self, *args, **kwargs):
        return self.__class__.integrated_autocorrelation_time(self,*args,**kwargs)


def _autocorrelation_1d(x):
    """Estimate the normalized autocorrelation function of a 1-D series

    Args:
        x: The series as a 1-D numpy array.

    Returns:
        array: The autocorrelation function of the time series.

    Taken from: https://github.com/dfm/emcee/blob/master/emcee/autocorr.py
    """
    from numpy import fft
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError('invalid dimensions for 1D autocorrelation function')

    n = 2**(2*len(x) - 1).bit_length()

    # Compute the FFT and then (from that) the auto-correlation function
    f = fft.fft(x,n=n)
    acf = fft.ifft(f * np.conjugate(f))[:len(x)].real

    acf /= acf[0]
    return acf
