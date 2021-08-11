"""Definition of :class:`Samples`, to hold products of likelihood sampling."""

import os
import re
import glob
import functools
import logging

import numpy as np

from cosmopipe.lib import utils, mpi
from cosmopipe.lib.catalog.base import BaseCatalog, vectorize_columns
from cosmopipe.lib.parameter import ParameterCollection, Parameter, ParamName

from .utils import nsigmas_to_quantiles_1d, nsigmas_to_quantiles_1d_sym, metrics_to_latex


class hybridmethod(object):
    """
    Descriptor that allows a class method to be called as instance method.
    Taken from https://stackoverflow.com/questions/28237955/same-name-for-classmethod-and-instancemethod.
    """
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

    """Class that holds samples drawn from likelihood."""
    logger = logging.getLogger('Samples')
    _broadcast_attrs = ['parameters','attrs','mpistate','mpiroot']

    @mpi.MPIInit
    def __init__(self, data=None, parameters=None, attrs=None):
        """
        Initialize :class:`Samples`.

        Parameters
        ----------
        data : dict, Samples
            Dictionary name: array.
            If :class:`Samples` instance, update ``self`` attributes.

        parameters : list, ParameterCollection, default=None
            Parameters.
            Defaults to ``data.keys()``.

        attrs : dict
            Other attributes.
        """
        if isinstance(data,Samples):
            self.__dict__.update(data.__dict__)
            return
        self.data = {}
        if parameters is None:
            parameters = list((data or {}).keys())
        if isinstance(parameters,ParameterCollection):
            self.parameters = parameters.copy()
        else:
            self.parameters = ParameterCollection(parameters)
        if data is not None:
            for name in data:
                self[name] = data[name]
        self.attrs = attrs or {}

    @classmethod
    def _multiple_columns(cls, column):
        """Whether ``column`` correspond to multiple columns (list or :class:`ParameterCollection`)."""
        return isinstance(column,(list,ParameterCollection))

    def set_default_parameter(self, name=None):
        """
        Add default parameter of name ``name``.

        Parameters
        ----------
        name : ParamName, string, tuple, Parameter
            Parameter name.
        """
        param = Parameter(name=name)
        if param.name.tuple[0] == 'metrics':
            param.latex = metrics_to_latex(param.name.tuple[1])
        self.parameters.setdefault(param)

    def columns(self, include=None, exclude=None, **kwargs):
        """
        Return parameter names, after optional selections.

        Parameters
        ----------
        include : list, string, default=None
            Single or list of *regex* patterns to select parameter names to include.
            Defaults to all parameters.

        exclude : list, string, default=None
            Single or list of *regex* patterns to select parameter names to exclude.
            Defaults to no parameters.

        kwargs : dict
            Selections on parameter attributes, e.g. ``varied=True`` for varied parameters.

        Returns
        -------
        columns : list
            Return parameters, after optional selections.
        """
        toret = super(Samples,self).columns(include=include,exclude=exclude)
        toret = [ParamName(name) for name in toret]

        if kwargs and self.is_mpi_root():
            parameter_selection = self.parameters.select(**kwargs)
            toret = [col for col in toret if col in parameter_selection]

        return self.mpicomm.bcast(toret,root=self.mpiroot)

    def get(self, name, *args, **kwargs):
        """
        Return samples for parameter ``name``.
        If not found, return default if provided.

        Parameters
        ----------
        name : ParamName, string, tuple, Parameter
            Parameter name.

        Returns
        -------
        samples : array
        """
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
        paramname = ParamName(name)
        if paramname not in self.data:
            for name in [('metrics','fweight'),('metrics','aweight')]:
                if name == paramname.tuple:
                    if self.is_mpi_gathered() and not self.is_mpi_root():
                        return None
                    return self.ones(dtype='f8')
            if paramname.tuple == ('metrics','weight'):
                fw, aw = self['metrics.fweight'], self['metrics.aweight']
                if fw is None and aw is None: return None
                return fw*aw
            if has_default:
                return default
        return self.data[paramname]

    def set(self, name, item):
        """
        Set parameter ``name`` samples to ``item``.

        Parameters
        ----------
        name : ParamName, string, tuple, Parameter
            Parameter name. If does not exist in current samples,
            creates new parameter.

        item : array
            Samples for this parameter.
        """
        self.data[ParamName(name)] = item
        self.set_default_parameter(name)

    def remove_burnin(self, burnin=0):
        """
        Return new samples with burn-in removed.

        Parameters
        ----------
        burnin : float, int
            If burnin between 0 and 1, remove that fraction of samples.
            Else, remove burnin first points (in global samples).

        Returns
        -------
        samples : Samples
        """
        if 0 < burnin < 1:
            burnin = burnin*self.gsize
        burnin = int(round(burnin))
        return self.gslice(burnin,None)

    def __contains__(self, name):
        """Whether samples exist for this parameter name ``name``."""
        return ParamName(name) in self.data

    def __getstate__(self):
        """Return this class state dictionary."""
        data = {str(name):col for name,col in self.data.items()}
        return {'data':data,'parameters':self.parameters.__getstate__(),'attrs':self.attrs}

    def __setstate__(self, state):
        """Set this class state dictionary."""
        self.data = state['data'].copy()
        self.parameters = ParameterCollection.from_state(state['parameters'])
        self.attrs = state['attrs']

    def __getitem__(self, name):
        """
        Get samples parameter ``name`` if :class:`Parameter`, :class:`ParamName`, string or tuple,
        else return copy with local slice of samples.
        """
        if isinstance(name,(Parameter,ParamName,str,tuple)):
            return self.get(name)
        new = self.copy()
        new.data = {col:self[col][name] for col in self.data}
        return new

    def __delitem__(self, name):
        """Delete samples for parameter ``name``."""
        del self.data[ParamName(name)]

    def __setitem__(self, name, item):
        """
        Set samples for parameter ``name`` if :class:`Parameter`, :class:`ParamName`, string or tuple,
        else set slice ``name`` of all columns to ``item``.
        """
        if isinstance(name,(Parameter,ParamName,str,tuple)):
            return self.set(name,item)
        for col in self.data:
            self[col][name] = item

    def __repr__(self):
        """Return string representation, including global size and columns."""
        return 'Samples(size={:d}, columns={})'.format(self.gsize,self.columns())

    def to_array(self, columns=None, struct=True):
        """
        Return samples as *numpy* array.

        Parameters
        ----------
        columns : list, default=None
            Columns to use. Defaults to all columns.

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
        columns = [col.name if isinstance(col,Parameter) else ParamName(col) for col in columns]
        if struct:
            toret = np.empty(self.size,dtype=[(str(col),self[col].dtype,self[col].shape[1:]) for col in columns])
            for col in columns: toret[str(col)] = self[col]
            return toret
        return np.array([self[col] for col in columns])

    def to_mesh(self, columns, **kwargs):
        """
        Interpolate samples to mesh.

        Parameters
        ----------
        columns : list, ParameterCollection
            List of parameters to build a mesh for.

        kwargs : dict
            Arguments for :meth:`Mesh.from_samples`.

        Returns
        -------
        mesh : Mesh
            Mesh with interpolated samples.
        """
        from .mesh import Mesh
        if columns is None: columns = self.columns(varied=True)
        if not self._multiple_columns(columns):
            columns = [columns]
        columns = list(columns) + ['metrics.weight']
        samples = []
        for col in columns: samples.append(self.gget(col))
        return Mesh.from_samples(samples[:-1],weights=samples[-1],dims=columns,**kwargs)

    @classmethod
    def load_auto(cls, filename, *args, **kwargs):
        """
        Load samples from disk.

        Parameters
        ----------
        filename : string
            File name of samples.
            If ends with '.txt', calls :meth:`load_cosmomc`.
            Else (numpy binary format), calls :meth:`load`.

        args : list
            Arguments for load function.

        kwargs : dict
            Other arguments for load function.
        """
        if os.path.splitext(filename)[-1] == '.txt':
            return cls.load_cosmomc(os.path.splitext(filename)[0],*args,**kwargs)
        return cls.load(filename,*args,**kwargs)

    def save_auto(self, filename, *args, **kwargs):
        """
        Save samples to disk.

        Parameters
        ----------
        filename : string
            File name of samples.
            If ends with '.txt', calls :meth:`load_cosmomc`.
            Else (*numpy* binary format), calls :meth:`save`.

        args : list
            Arguments for save function.

        kwargs : dict
            Other arguments for save function.
        """
        if os.path.splitext(filename)[-1] == '.txt':
            return self.save_cosmomc(os.path.splitext(filename)[0],*args,**kwargs)
        return self.save(filename)

    @classmethod
    @mpi.CurrentMPIComm.enable
    def load_cosmomc(cls, base_filename, ichains=None, mpiroot=0, mpistate=mpi.CurrentMPIState.GATHERED, mpicomm=None):
        """
        Load samples in *CosmoMC* format, i.e.:

        - '_{ichain}.txt' files for sample values
        - '.paramnames' files for parameter names / latex
        - '.ranges' for parameter ranges

        Parameters
        ----------
        base_filename : string
            Base *CosmoMC* file name. Will be prepended by '_{ichain}.txt' for sample values,
            '.paramnames' for parameter names and '.ranges' for parameter ranges.

        ichains : int, tuple, list, default=None
            Chain numbers to load. Defaults to all chains matching pattern '{base_filename}*.txt'

        mpiroot : int, default=0
            Rank of root process.

        mpistate : string, mpi.CurrentMPIState
            MPI state: 'scattered', 'gathered', 'broadcast'?

        mpicomm : MPI communicator, default=None
            MPI communicator.

        Returns
        -------
        samples : Samples
        """
        self = cls(mpiroot=mpiroot,mpistate=mpi.CurrentMPIState.GATHERED,mpicomm=mpicomm)

        if self.is_mpi_root():
            parameters_filename = '{}.paramnames'.format(base_filename)
            self.log_info('Loading parameters file: {}.'.format(parameters_filename))
            self.parameters = ParameterCollection()
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
                    chain_filenames.append(chain_filename.format('_{:d}'.format(ichain)))
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
        """
        Save samples to disk in *CosmoMC* format.

        Parameters
        ----------
        base_filename : string
            Base *CosmoMC* file name. Will be prepended by '_{ichain}.txt' for sample values,
            '.paramnames' for parameter names and '.ranges' for parameter ranges.

        columns : list, ParameterCollection, default=None
            Parameters to save samples of. Defaults to all parameters (weight and logposterior treated separatey).

        ichain : int, default=None
            Chain number to append to file name, i.e. sample values will be saved as '{base_filename}_{ichain}.txt'.
            If ``None``, does not append any number, sample values will be saved as '{base_filename}.txt'.

        kwargs : dict
            Arguments for :func:`numpy.savetxt`.
        """
        isscattered = self.is_mpi_scattered()
        if isscattered: self.mpi_gather()
        if columns is None: columns = self.columns(exclude='metrics.*')
        else: columns = list(columns)
        if self.is_mpi_root():
            metrics_columns = ['metrics.weight','metrics.logposterior']
            for column in metrics_columns:
                if column in columns: del columns[columns.index(column)]
            data = self.to_array(columns=metrics_columns + columns,struct=False)
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
        """
        Return *GetDist* hook to samples.

        Parameters
        ----------
        columns : list, ParameterCollection, default=None
            Parameters to share to *GetDist*. Defaults to all parameters (weight and logposterior treated separatey).

        Returns
        -------
        samples : getdist.MCSamples
        """
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

    def cov(self, columns=None, ddof=1):
        """
        Estimate weighted parameter covariance.

        Parameters
        ----------
        columns : list, ParameterCollection, default=None
            Parameters to compute covariance for.
            Defaults to all varied parameters.

        ddof : int, default=1
            Number of degrees of freedom.

        Returns
        -------
        cov : scalar, array
            If single parameter provided as ``columns``, returns variance for that parameter (scalar).
            Else returns covariance (2D array).
        """
        return super(Samples,self).cov(columns,fweights=self['metrics.fweight'],aweights=self['metrics.aweight'],ddof=ddof)


    def invcov(self, columns=None, ddof=1):
        """
        Estimate weighted parameter inverse covariance.

        Parameters
        ----------
        columns : list, ParameterCollection, default=None
            Parameters to compute inverse covariance for.
            Defaults to all varied parameters.

        ddof : int, default=1
            Number of degrees of freedom.

        Returns
        -------
        cov : scalar, array
            If single parameter provided as ``columns``, returns inverse variance for that parameter (scalar).
            Else returns inverse covariance (2D array).
        """
        return utils.inv(self.cov(columns,ddof=1))

    def corrcoef(self, columns=None, **kwargs):
        """
        Estimate weighted parameter correlation matrix.
        See :meth:`cov`.
        """
        return utils.cov_to_corrcoef(self.cov(columns,**kwargs))

    def covpair(self, column1, column2, **kwargs):
        """
        Estimate weighted covariance of a pair of parameters ``column1`` and ``column2``.
        See :meth:`cov`.
        """
        return self.cov([column1,column2],**kwargs)[0,1]

    def corrpair(self, column1, column2, **kwargs):
        """
        Estimate weighted correlation of a pair of parameters ``column1`` and ``column2``.
        See :meth:`cov`.
        """
        return self.corrcoef([column1,column2],**kwargs)[0,1]

    @vectorize_columns
    def var(self, column, ddof=1):
        """
        Estimate weighted parameter variance.

        Parameters
        ----------
        columns : list, ParameterCollection, default=None
            Parameters to compute variance for.

        ddof : int, default=1
            Number of degrees of freedom.

        Returns
        -------
        var : scalar, array
            If single parameter provided as ``columns``, returns variance for that parameter (scalar).
            Else returns variance array.
        """
        return super(Samples,self).var(column,fweights=self['metrics.fweight'],aweights=self['metrics.aweight'],ddof=ddof)

    @vectorize_columns
    def mean(self, column):
        """Return weighted mean."""
        return super(Samples,self).average(column,weights=self['metrics.weight'])

    @vectorize_columns
    def argmin(self, column, cost='metrics.chi2'):
        """Return parameter value for minimum of ``cost.``"""
        if self.is_mpi_scattered():
            argmin,rank = mpi.argmin_array(self[cost],mpicomm=self.mpicomm)
            return self.mpicomm.bcast(self[column][argmin] if self.mpicomm.rank == rank else None,root=rank)
        isroot = self.is_mpi_root()
        return self.mpicomm.bcast(self[column][np.argmin(self[cost])] if isroot else None,root=self.mpiroot)

    @vectorize_columns
    def argmax(self, column, cost='metrics.logposterior'):
        """Return parameter value for maximum of ``cost.``"""
        if self.is_mpi_scattered():
            argmax,rank = mpi.argmax_array(self[cost],mpicomm=self.mpicomm)
            return self.mpicomm.bcast(self[column][argmax] if self.mpicomm.rank == rank else None,root=rank)
        isroot = self.is_mpi_root()
        return self.mpicomm.bcast(self[column][np.argmax(self[cost])] if isroot else None,root=self.mpiroot)

    @vectorize_columns
    def quantile(self, column, q=(0.1587,0.8413)):
        """Return weighted quantiles."""
        return super(Samples,self).quantile(column=column,q=q,weights=self['metrics.weight'])

    @vectorize_columns
    def interval(self, column, nsigmas=1., bins=100, method='gaussian_kde', bw_method='scott'):
        """
        Return n-sigmas confidence interval(s).

        Parameters
        ----------
        columns : list, ParameterCollection, default=None
            Parameters to compute confidence interval for.

        nsigmas : int
            Return interval for this number of sigmas.

        bins : int, default=100
            Number of bins i.e. mesh nodes.
            See :meth:`Mesh.from_samples`.

        method : string
            Method to interpolate (weighted) samples on mesh.
            See :meth:`Mesh.from_samples`.

        bw_method : string, default='scott'
            If ``method`` is ``'gaussian_kde'``, method to determine KDE bandwidth, see :class:`scipy.stats.gaussian_kde`.

        Returns
        -------
        interval : array
        """
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
        """
        Export samples summary quantities.

        Parameters
        ----------
        columns : list, default=None
            Parameters to export quantities for.
            Defaults to all parameters.

        quantities : list, default=None
            Quantities to export. Defaults to ``['argmax','mean','median','std','quantile:1sigma','interval:1sigma']``.

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
        if columns is None: columns = self.columns(varied=True)
        parameters = [self.parameters[col] for col in columns]
        data = []
        if quantities is None: quantities = ['argmax','mean','median','std','quantile:1sigma','interval:1sigma']
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
                if quantity in ['argmax','mean','median','std']:
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
        Return Gelman-Rubin statistics, which compares covariance of chain means to (mean of) intra-chain covariances.

        Parameters
        ----------
        chains : list
            List of :class:`Samples` instances.

        columns : list, ParameterCollection
            Parameters to compute Gelman-Rubin statistics for.
            Defaults to all parameters.

        statistic : string, callable, default='mean'
            If 'mean', compares covariance of chain means to (mean of) intra-chain covariances.
            Else, must be a callable taking :class:`Samples` instance and parameter list as input
            and returning array of values (one for each parameter).

        method : string, default='eigen'
            If `eigen`, return eigenvalues of covariance ratios, else diagonal.

        return_matrices : bool, default=True
            If ``True``, also return pair of covariance matrices.

        check : bool, default=True
            Whether to check for reliable inverse of intra-chain covariances.

        Reference
        ---------
        http://www.stat.columbia.edu/~gelman/research/published/brooksgelman2.pdf
        """
        if columns is None: columns = chains[0].columns(varied=True)
        isscalar = not cls._multiple_columns(columns)
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
    def autocorrelation(self, column):
        """
        Return weighted autocorrelation.
        Adapted from https://github.com/dfm/emcee/blob/main/src/emcee/autocorr.py

        Parameters
        ----------
        columns : list, ParameterCollection
            Parameters to compute autocorrelation for.
            Defaults to all parameters.

        Returns
        -------
        autocorr : array
        """
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
        if self.is_mpi_scattered():
            return mpi.scatter_array(autocorr,root=self.mpiroot,mpicomm=self.mpicomm)
        if self.is_mpi_broadcast():
            return mpi.broadcast_array(autocorr,root=self.mpiroot,mpicomm=self.mpicomm)
        return autocorr

    @hybridmethod
    def integrated_autocorrelation_time(cls, chains, column, min_corr=None, c=5, reliable=50, check=False):
        """
        Return integrated autocorrelation time.
        Adapted from https://github.com/dfm/emcee/blob/main/src/emcee/autocorr.py

        Parameters
        ----------
        chains : list
            List of :class:`Samples` instances.

        columns : list, ParameterCollection
            Parameters to compute integrated autocorrelation time for.

        min_corr : float, default=None
            Integrate starting from this lower autocorrelation threshold.
            If ``None``, use ``c``.

        c : float, int
            Step size for the window search.

        reliable : float, int, default=50
            Minimum ratio between the chain length and estimated autocorrelation time
            for it to be considered reliable.

        check : bool, default=False
            Whether to check for reliable estimate of autocorrelation time (based on ``reliable``).

        Returns
        -------
        iat : scalar, array
        """
        if not isinstance(chains,(list,tuple)):
            chains = [chains]

        if cls._multiple_columns(column):
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
            corr_ = samples.autocorrelation(column)
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
        """Instance method corresponding to class method :meth:`integrated_autocorrelation_time`."""
        return self.__class__.integrated_autocorrelation_time(self,*args,**kwargs)


def _autocorrelation_1d(x):
    """
    Estimate the normalized autocorrelation function.
    Taken from https://github.com/dfm/emcee/blob/main/src/emcee/autocorr.py

    Parameters
    ----------
    x : array
        1D time series.

    Returns
    -------
    acf : array
        The autocorrelation function of the time series.
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
