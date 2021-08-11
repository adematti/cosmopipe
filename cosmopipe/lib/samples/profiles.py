"""Definition of :class:`Profiles`, to hold products of likelihood profiling."""

import os
import logging

import numpy as np

from cosmopipe.lib.utils import BaseClass
from cosmopipe.lib import utils
from cosmopipe.lib.parameter import ParameterCollection, Parameter, ParamName


class ParamDict(BaseClass):

    """Class for a simple name: item mapping."""

    def __init__(self, data):
        """
        Initialize :class:`ParamDict`.

        Parameters
        ----------
        data : dict, ParamDict
            Dictionary name: array.
            If :class:`ParamDict` instance, update ``self`` attributes.
        """
        if isinstance(data,ParamDict):
            self.__dict__.update(data.__dict__)
            return
        self.data = {}
        for key,value in data.items():
            self[key] = value

    def columns(self):
        """Return parameter names."""
        return list(self.data.keys())

    def __getitem__(self, name):
        """Get item for parameter ``name``."""
        return self.data[ParamName(name)]

    def __setitem__(self, name, item):
        """Set item for parameter ``name``."""
        self.data[ParamName(name)] = item

    def __getstate__(self):
        """Return this class state dictionary."""
        data = {str(name):col for name,col in self.data.items()}
        return {'data':data}

    def __setstate__(self, state):
        """Set this class state dictionary."""
        self.data = {ParamName(name):col for name,col in state['data'].items()}

    def items(self):
        """Return dictionary items, i.e. (name, item) tuples."""
        return self.data.items()

    def __repr__(self):
        """Return string representation of catalog, including internal data."""
        return '{}({})'.format(self.__class__.__name__,self.data)

    def __contains__(self, name):
        """Whether there is an entry for parameter ``name``."""
        return ParamName(name) in self.data


class ParamCovariance(BaseClass):
    """
    Class that represents a parameter covariance.
    TODO: think about link with data_vector.CovarianceMatrix and mixing compressed data data vector covariance / parameter covariance, e.g. pre-recon + alpha?
    """
    def __init__(self, covariance, parameters):
        """
        Initialize :class:`ParamCovariance`.

        Parameters
        ----------
        covariance : array
            2D array holding covariance.

        parameters : list, ParameterCollection
            Parameters corresponding to input ``covariance``.
        """
        self._cov = np.asarray(covariance)
        self.parameters = ParameterCollection(parameters)

    def cov(self, parameters=None):
        """Return covariance matrix for input parameters ``parameters``."""
        if parameters is None:
            parameters = self.parameters
        idx = np.array([self.parameters.index(param) for param in parameters])
        toret = self._cov[np.ix_(idx,idx)]
        return toret

    def invcov(self, parameters=None):
        """Return inverse covariance matrix for input parameters ``parameters``."""
        return utils.inv(self.cov(parameters))

    def corrcoef(self, parameters=None):
        """Return correlation matrix for input parameters ``parameters``."""
        return utils.cov_to_corrcoef(self.cov(parmeters=parameters))

    def __getstate__(self):
        """Return this class state dictionary."""
        return {'cov':self._cov,'parameters':self.parameters.__getstate__()}

    def __setstate__(self, state):
        """Set this class state dictionary."""
        self._cov = state['cov']
        self.parameters = ParameterCollection.from_state(state['parameters'])

    def __repr__(self):
        """Return string representation of parameter covariance, including parameters."""
        return '{}({})'.format(self.__class__.__name__,self.parameters)


class Profiles(BaseClass):
    r"""
    Class holding results of likelihood profiling.

    Attributes
    ----------
    init : ParamDict
        Initial parameter values.

    bestfit : ParamDict
        Best fit parameters.

    parabolic_errors : ParamDict
        Parameter parabolic errors.

    deltachi2_errors : ParamDict
        Lower and upper errors corresponding to :math:`\Delta \chi^{2} = 1`.

    covariance : ParamCovariance
        Parameter covariance at best fit.
    """
    logger = logging.getLogger('Profiles')
    _paramdicts = ['metrics','init','bestfit','parabolic_errors','deltachi2_errors']

    def __init__(self, parameters, attrs=None):
        """
        Initialize :class:`Profiles`.

        Parameters
        ----------
        parameters : list, ParameterCollection
            Parameters used in likelihood profiling.

        attrs : dict, default=None
            Other attributes.
        """
        self.parameters = ParameterCollection(parameters)
        self.attrs = attrs or {}

    def set_metrics(self, values):
        """Set metrics (e.g. 'loglkl') dictionary ``values``."""
        self.metrics = ParamDict(values)

    def set_init(self, values):
        """Set initial parameter values ``values``."""
        self.init = ParamDict(values)

    def set_bestfit(self, values):
        """Set best fit parameter values ``values``."""
        self.bestfit = ParamDict(values)

    def set_parabolic_errors(self, values):
        """Set parameter parabolic errors ``values``."""
        self.parabolic_errors = ParamDict(values)

    def set_deltachi2_errors(self, values):
        r"""Set parameter :math:`\Delta \chi^{2} = 1` errors ``values``."""
        self.deltachi2_errors = ParamDict(values)

    def set_covariance(self, covariance, parameters=None):
        """Set parameter covariance ``covariance`` for parameters ``parameters``."""
        self.covariance = ParamCovariance(covariance,parameters=self.parameters if parameters is None else parameters)

    def argmin(self):
        """Return index of best best fit among all the tries."""
        return self.metrics['fval'].argmin()

    def ntries(self):
        """Return number of tries."""
        return len(self.metrics['fval'])

    def __len__(self):
        """Length is number of tries."""
        return self.ntries()

    def get(self, name):
        """Access attribute by name."""
        return getattr(self,name)

    def has(self, name):
        """Has this attribute?"""
        return hasattr(self,name)

    @classmethod
    def concatenate(cls, *others):
        """
        Concatenate profiles together.

        Parameters
        ----------
        others : list
            List of :class:`Profiles` instances.

        Returns
        -------
        new : Profiles

        Warning
        -------
        :attr:`attrs` of returned profiles contains, for each key, the last value found in ``others`` :attr:`attrs` dictionaries.
        """
        new = others[0].copy()
        new_ntries = new.ntries()
        for other in others:
            for name in ['init','metrics','bestfit','parabolic_errors']:
                if hasattr(new,name) and hasattr(other,name):
                    setattr(new,name,np.concatenate([getattr(new,name),getattr(other,name)],axis=0))
                elif hasattr(new,name) ^ hasattr(other,name):
                    raise ValueError('Cannot append two profiles if both do not have {}.'.format(name))
            for name in ['covariance','deltachi2_errors']:
                if new.argmin() >= new_ntries:
                    if not hasattr(other,name):
                        raise ValueError('{} not provided for the global bestfit.'.format(name))
                    setattr(new,name,getattr(other,name).copy())
        return new

    def extend(self, other):
        """Extend profiles with ``other``."""
        new = self.concatenate(self,other)
        self.__dict__.update(new.__dict__)

    def __getstate__(self):
        """Return this class state dictionary."""
        state = {}
        state['attrs'] = self.attrs
        for name in ['parameters'] + self._paramdicts + ['covariance']:
            if hasattr(self,name):
                state[name] = getattr(self,name).__getstate__()
        return state

    def __setstate__(self, state):
        """Set this class state dictionary."""
        self.parameters = ParameterCollection.from_state(state['parameters'])
        self.attrs = state['attrs']
        for name in self._paramdicts:
            if name in state:
                setattr(self,name,ParamDict.from_state(state[name]))
        if 'covariance' in state:
            self.covariance = ParamCovariance.from_state(state['covariance'])

    def __repr__(self):
        """Return string representation of profiles, including parameters."""
        return '{}({})'.format(self.__class__.__name__,self.parameters)

    @classmethod
    def to_samples(cls, profiles, parameters=None, name='bestfit', select='min'):
        """
        Export profiles as :class:`Samples`.
        This is not statistically meaningful, but convenient to share plotting routines.

        Parameters
        ----------
        profiles : list
            List of :class:`Profiles` instances.

        parameters : list, ParameterCollection, default=None
            Parameters to export. Defaults to all parameters.

        name : string, default='bestfit'
            Quantity to export as samples.

        select : string, default='min'
            Rule to select the quantity to export.

        Returns
        -------
        samples : Samples
        """
        from .samples import Samples
        if parameters is None: parameters = profiles[0].parameters
        parameters = [profiles[0].parameters[param] for param in parameters]
        samples = Samples(parameters=parameters,mpicomm=profiles[0].mpicomm)
        for param in parameters:
            if select == 'min':
                samples[param] = np.array([prof.get(name)[param][prof.argmin()] for prof in profiles])
                for metrics in ['loglkl','logprior','logposterior']:
                    if all(metrics in prof.metrics for prof in profiles):
                        samples['metrics.{}'.format(metrics)] = np.array([prof.metrics[metrics][prof.argmin()] for prof in profiles])
            else:
                raise NotImplementedError('Cannot export profiles to samples with select = {}'.format(select))
        return samples


    def to_stats(self, parameters=None, quantities=None, sigfigs=2, tablefmt='latex_raw', filename=None):
        """
        Export profiling quantities.

        Parameters
        ----------
        parameters : list, ParameterCollection
            Parameters to export quantities for.
            Defaults to all parameters.

        quantities : list, default=None
            Quantities to export. Defaults to ``['bestfit','parabolic_errors','deltachi2_errors']``.

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
        if parameters is None: parameters = self.parameters
        data = []
        if quantities is None: quantities = [quantity for quantity in ['bestfit','parabolic_errors','deltachi2_errors'] if self.has(quantity)]
        is_latex = 'latex_raw' in tablefmt
        argmin = self.argmin()

        def round_errors(low, up):
            low,up = utils.round_measurement(0.0,low,up,sigfigs=sigfigs)[1:]
            if is_latex: return '${{}}_{{{}}}^{{+{}}}$'.format(low,up)
            return '{}/+{}'.format(low,up)

        for iparam,param in enumerate(parameters):
            row = []
            if is_latex: row.append(param.get_label())
            else: row.append(str(param.name))
            row.append(str(param.varied))
            ref_error = self.parabolic_errors[param][argmin]
            for quantity in quantities:
                if quantity in ['bestfit','parabolic_errors']:
                    value = self.get(quantity)[param][argmin]
                    value = utils.round_measurement(value,ref_error,sigfigs=sigfigs)[0]
                    if is_latex: value = '${}$'.format(value)
                    row.append(value)
                elif quantity == 'deltachi2_errors':
                    low,up = self.get(quantity)[param][argmin]
                    row.append(round_errors(low,up))
                else:
                    raise RuntimeError('Unknown quantity {}.'.format(quantity))
            data.append(row)
        headers = []
        if 'loglkl' in self.metrics:
            chi2min = '{:.2f}'.format(-2.*self.metrics['loglkl'][argmin])
            headers.append(('$\chi^{{2}} = {}$' if is_latex else 'chi2 = {}').format(chi2min))
        headers.append('varied')
        headers += [quantity.replace('_',' ') for quantity in quantities]
        tab = tabulate.tabulate(data,headers=headers,tablefmt=tablefmt)
        if filename and self.is_mpi_root():
            utils.mkdir(os.path.dirname(filename))
            self.log_info('Saving to {}.'.format(filename),rank=0)
            with open(filename,'w') as file:
                file.write(tab)
        return tab
