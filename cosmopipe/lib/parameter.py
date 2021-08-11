"""Classes to handle parameters."""

import sys
import logging
import math
import re
import fnmatch
import itertools

import numpy as np
from scipy import stats
from pypescript.syntax import Decoder

from . import utils
from .utils import BaseClass, BaseOrderedCollection
from . import mpi


def decode_name(name, default_start=0, default_stop=None, default_step=1):
    """
    Split ``name`` into strings and allowed index ranges.

    >>> decode_name('a_[-4:5:2]_b_[0:2]')
    ['a_','_b_'], [range(-4,5,2),range(0,2,1)]

    Parameters
    ----------
    name : string
        Parameter name, e.g. ``a_[-4:5:2]``.

    default_start : int, default=0
        Range start to use as a default.

    default_stop : int, default=None
        Range stop to use as a default.

    default_step : int, default=1
        Range step to use as a default.

    Returns
    -------
    strings : list
        List of strings.

    ranges : list
        List of ranges.
    """
    name = str(ParamName(name))
    replaces = re.finditer('\[(-?\d*):(\d*):*(-?\d*)\]',name)
    strings, ranges = [], []
    string_start = 0
    for ireplace,replace in enumerate(replaces):
        start, stop, step = replace.groups()
        if not start:
            start = default_start
            if start is None:
                raise ValueError('You must provide a lower limit to parameter index')
        else: start = int(start)
        if not stop:
            stop = default_stop
            if stop is None:
                raise ValueError('You must provide an upper limit to parameter index')
        else: stop = int(stop)
        if not step:
            step = default_step
            if step is None:
                raise ValueError('You must provide a step for parameter index')
        else: step = int(step)
        strings.append(name[string_start:replace.start()])
        string_start = replace.end()
        ranges.append(range(start,stop,step))

    strings += [name[string_start:]]

    return strings, ranges


def yield_names_latex(name, latex=None, **kwargs):
    r"""
    Yield parameter name and latex strings with template forms ``[::]`` replaced.

    >>> yield_names_latex('a_[-4:3:2]',latex='\alpha_[-4:5:2]')
    a_-4, \alpha_{-4}
    a_-2, \alpha_{-2}
    a_-0, \alpha_{-0}
    a_2, \alpha_{-2}

    Parameters
    ----------
    name : string
        Parameter name.

    latex : string, default=None
        Latex for parameter.

    kwargs : dict
        Arguments for :func:`decode_name`

    Returns
    -------
    name : string
        Parameter name with template forms ``[::]`` replaced.

    latex : string, None
        If input ``latex`` is ``None``, ``None``.
        Else latex string with template forms ``[::]`` replaced.
    """
    strings,ranges = decode_name(name,**kwargs)

    if not ranges:
        yield strings[0], latex

    else:
        import itertools

        template = '{:d}'.join(strings)
        if latex is not None:
            latex = latex.replace('[]','{{{:d}}}')

        for nums in itertools.product(*ranges):
            yield template.format(*nums), latex.format(*nums) if latex is not None else latex


def find_names_latex(allnames, name, latex=None, quiet=True):
    r"""
    Search parameter name ``name`` in list of names ``allnames``,
    matching template forms ``[::]``;
    return corresponding parameter names and latex.

    >>> find_names_latex(['a_1','a_2','b_1'],'a_[:]',latex='\alpha_[:]')
    [('a_1','\alpha_{1}'), ('a_2','\alpha_{2}')]

    Parameters
    ----------
    allnames : list
        List of parameter names (strings).

    name : string
        Parameter name to match in ``allnames``.

    latex : string, default=None
        Latex for parameter.

    quiet : bool, default=True
        If ``False`` and no match for ``name`` was found is ``allnames``, raise :class:`ParamError`.

    Returns
    -------
    toret : list
        List of string tuples ``(name, latex)``.
        ``latex`` is ``None`` if input ``latex`` is ``None``.
    """
    name = str(ParamName(name))
    error = ParamError('No match found for {}'.format(name))
    strings,ranges = decode_name(name,default_start=-sys.maxsize,default_stop=sys.maxsize)
    if not ranges:
        if strings[0] in allnames:
            return [(strings[0], latex)]
        if not quiet:
            raise error
        return []
    pattern = re.compile('(-?\d*)'.join(strings))
    toret = []
    if latex is not None:
        latex = latex.replace('[]','{{{:d}}}')
    for paramname in allnames:
        match = re.match(pattern,paramname)
        if match:
            add = True
            nums = []
            for s,ra in zip(match.groups(),ranges):
                idx = int(s)
                nums.append(idx)
                add = idx in ra # ra not in memory
                if not add: break
            if add:
                toret.append((paramname,latex.format(*nums) if latex is not None else latex))
    if not toret and not quiet:
        raise error
    return toret


def find_names(allnames, name, quiet=True):
    """
    Search parameter name ``name`` in list of names ``allnames``,
    matching template forms ``[::]``;
    return corresponding parameter names.
    Contrary to :func:`find_names_latex`, it does not handle latex strings,
    but can take a list of parameter names as ``name``
    (thus returning the concatenated list of matching names in ``allnames``).

    >>> find_names(['a_1','a_2','b_1','c_2'],['a_[:]','b_[:]'])
    ['a_1','a_2','b_1']

    Parameters
    ----------
    allnames : list
        List of parameter names (strings).

    name : list, string
        List of parameter name(s) to match in ``allnames``.

    quiet : bool, default=True
        If ``False`` and no match for parameter name was found is ``allnames``, raise :class:`ParamError`.

    Returns
    -------
    toret : list
        List of parameter names (strings).
    """
    if isinstance(name,list):
        toret = []
        for name_ in name: toret += find_names(allnames,name_,quiet=quiet)
        return toret

    name = str(ParamName(name))
    error = ParamError('No match found for {}'.format(name))

    name = fnmatch.translate(name)
    strings,ranges = decode_name(name,default_start=-sys.maxsize,default_stop=sys.maxsize)
    pattern = re.compile('(-?\d*)'.join(strings))
    toret = []
    for paramname in allnames:
        match = re.match(pattern,paramname)
        if match:
            add = True
            nums = []
            for s,ra in zip(match.groups(),ranges):
                idx = int(s)
                nums.append(idx)
                add = idx in ra # ra not in memory
                if not add: break
            if add:
                toret.append(paramname)
    if not toret and not quiet:
        raise error
    return toret


class ParamError(Exception):

    """Exception raised when issue with :class:`ParamError`."""


class ParameterCollection(BaseOrderedCollection):

    """Class holding a collection of parameters."""
    logger = logging.getLogger('ParameterCollection')

    def __init__(self, data=None, parser=None):
        """
        Initialize :class:`ParameterCollection`.

        Parameters
        ----------
        data : list, tuple, string, dict, ParameterCollection
            Can be:

            - list (or tuple) of parameters (:class:`Parameter` or dictionary to initialize :class:`Parameter`).
            - path to a configuration *yaml* file to decode
            - dictionary of name: parameter
            - ParameterCollection instance

        string : string
            If not ``None``, *yaml* format string to decode.
            Added on top of ``data``.

        parser : callable, default=yaml_parser
            Function that parses *yaml* string into a dictionary.
            Used when ``data`` is string, or ``string`` is not ``None``.
        """
        if isinstance(data,ParameterCollection):
            self.__dict__.update(data.__dict__)
            return

        self.data = []
        if isinstance(data,(list,tuple)):
            data_ = data
            data = {}
            for name in data_:
                if isinstance(name,Parameter):
                    data[name.name] = name
                elif isinstance(name,dict):
                    data[name['name']] = name
                else:
                    data[name] = {}

        elif not isinstance(data,dict):
            data = Decoder(data=data,parser=parser)

        for name,conf in data.items():
            if isinstance(conf,Parameter):
                self.set(conf)
            else:
                latex = conf.pop('latex',None)
                for name,latex in yield_names_latex(name,latex=latex):
                    param = Parameter(name=name,latex=latex,**conf)
                    self.set(param)
                    conf = conf.copy()

    def get(self, name):
        """
        Return parameter of name ``name`` in collection.

        Parameters
        ----------
        name : Parameter, ParamName, string, tuple
            Parameter name.
            If :class:`Parameter` instance, search for parameter with same name.

        Returns
        -------
        param : Parameter
        """
        return self.data[self.index(name)]


    def __setitem__(self, name, item):
        """
        Update parameter in collection (a parameter with same name must already exist).
        See :meth:`set` to set a new parameter.

        Parameters
        ----------
        name : Parameter, ParamName, string, tuple, int
            Parameter name.
            If :class:`Parameter` instance, search for parameter with same name.
            If integer, index in collection.

        item : Parameter
            Parameter.
        """
        if not isinstance(item,Parameter):
            raise TypeError('{} is not a Parameter instance.'.format(item))
        try:
            self.data[name] = item
        except TypeError:
            if isinstance(name,Parameter):
                name = name.name
            name = ParamName(name)
            if item.name != name:
                raise KeyError('Parameter {} should be indexed by name (incorrect {})'.format(item.name,name))
            self.data[self._index_name(name)] = item

    def __getitem__(self, name):
        """
        Return parameter ``name``.

        Parameters
        ----------
        name : Parameter, ParamName, string, tuple, int
            Parameter name.
            If :class:`Parameter` instance, search for parameter with same name.
            If integer, index in collection.

        Returns
        -------
        param : Parameter
        """
        try:
            return self.data[name]
        except TypeError:
            return self.data[self.index(name)]

    def __delitem__(self, name):
        """
        Delete parameter ``name``.

        Parameters
        ----------
        name : Parameter, ParamName, string, tuple, int
            Parameter name.
            If :class:`Parameter` instance, search for parameter with same name.
            If integer, index in collection.
        """
        try:
            del self.data[name]
        except TypeError:
            del self.data[self.index(name)]

    def set(self, param):
        """
        Set parameter ``param`` in collection.
        If there is already a parameter with same name in collection, replace this stored parameter by the input one.
        Else, append parameter to collection.
        """
        if not isinstance(param,Parameter):
            raise TypeError('{} is not a Parameter instance.'.format(param))
        if param in self:
            self[param.name] = param
        else:
            self.data.append(param)

    def names(self):
        """Return names of parameters (:class:`ParamName` instances) in collection."""
        return (item.name for item in self.data)

    def index(self, name):
        """
        Return index of parameter ``name``.

        Parameters
        ----------
        name : Parameter, ParamName, string, tuple, int
            Parameter name.
            If :class:`Parameter` instance, search for parameter with same name.
            If integer, index in collection.

        Returns
        -------
        index : int
        """
        if isinstance(name,Parameter):
            name = name.name
        name = ParamName(name)
        return self._index_name(name)

    def _index_name(self, name):
        # get index of parameter name ``name``
        return list(self.names()).index(name)

    def __contains__(self, name):
        """Whether collection contains parameter ``name``."""
        if isinstance(name,Parameter):
            return name.name in self.names()
        return ParamName(name) in self.names()

    def setdefault(self, param):
        """Set parameter ``param`` in collection if not already in it."""
        if not isinstance(param,Parameter):
            raise TypeError('{} is not a Parameter instance.'.format(param))
        if param.name not in self:
            self.set(param)

    def __setstate__(self, state):
        """Set this class state dictionary."""
        self.data = [Parameter.from_state(param) for param in state['data']]

    def select(self, **kwargs):
        """
        Return new collection, after selection of parameters whose attribute match input values::

            collection.select(fixed=True)

        returns collection of fixed parameters.
        If 'name' is provided, consider all matching parameters, e.g.::

            collection.select(varied=True,name='a_[0:2]')

        returns a collection of varied parameters, with name in ``['a_0', 'a_1']``.
        """
        toret = self.__class__()
        name = kwargs.pop('name',None)
        if name is not None:
            names = find_names(map(str,self.names()),name)
            if not names: return toret # no match
        else:
            names = self.names()
        for name in names:
            param = self[name]
            if all(getattr(param,key) == val for key,val in kwargs.items()):
                toret.set(param)
        return toret


class ParamName(BaseClass):
    """
    Class representing a parameter name.

    Attributes
    ----------
    tuple : tuple
        Tuple of string, typically (section, name), e.g. ``('galaxy_bias','b1')``.
    """
    sep = '.'

    def __init__(self, *names):
        """
        Initialize :class:`ParamName`.

        Parameters
        ----------
        names : tuple, list, string, Parameter, ParamName
            Can be:

            - tuple or list of strings, e.g. ``('galaxy_bias','b1')``
            - string with sections separated by a dot, e.g. ``'galaxy_bias.b1'``
            - :class:`Parameter` instance, in which case its ``name`` attribute is copied.
            - :class:`ParamName` instance, which is copied.
        """
        if len(names) == 1:
            if isinstance(names[0],Parameter):
                self.__dict__.update(names[0].name.__dict__)
                return
            if isinstance(names[0],self.__class__):
                self.__dict__.update(names[0].__dict__)
                return
            if isinstance(names[0],str):
                names = tuple(names[0].split(self.sep))
            if isinstance(names[0],(tuple,list)):
                names = tuple(names[0])
        self.tuple = tuple(str(name) for name in names)

    def add_suffix(self, suffix):
        """Add suffix to parameter name, i.e. append '_suffix'."""
        self.tuple = self.tuple[:-1] + ('{}_{}'.format(self.tuple[-1],suffix),)

    def __repr__(self):
        """Represent parameter name as a string."""
        return '{}{}'.format(self.__class__.__name__,self.tuple)

    def __str__(self):
        """Return parameter name as a string."""
        return self.sep.join(self.tuple)

    def __eq__(self, other):
        """
        Is ``self`` equal to ``other``?
        Other can be:

        - string (checks it matches :meth:`__str__`)
        - tuple (checks it matches :attr:`tuple`)
        - :class:`ParamName` (checks :attr:`tuple` match)

        Note
        ----
        Behaviour is asymetric.

        >>> ParamName('galaxy_bias.b1') == 'galaxy_bias.b1'
        True
        >>> 'galaxy_bias.b1' == ParamName('galaxy_bias.b1')
        False
        """
        if isinstance(other,str):
            return other == str(self)
        if isinstance(other,tuple):
            return other == self.tuple
        return isinstance(other,self.__class__) and other.tuple == self.tuple

    def __hash__(self):
        """Hash parameter name."""
        return hash(str(self))

    def __setstate__(self, state):
        """Set the class state dictionary."""
        self.tuple = state['data']

    def __getstate__(self):
        """Return this class state dictionary."""
        return {'data':self.tuple}


class Parameter(BaseClass):
    """
    Class that represents a parameter.

    Attributes
    ----------
    name : ParamName
        Parameter name.

    value : float
        Default value for parameter.

    fixed : bool
        Whether parameter is fixed.

    prior : Prior
        Prior distribution.

    ref : Prior
        Reference distribution.
        This is supposed to represent the expected posterior for this parameter.

    proposal : float
        Proposal uncertainty.

    latex : string, default=None
        Latex for parameter.
    """
    _attrs = ['name','value','fixed','prior','ref','proposal','latex']
    logger = logging.getLogger('Parameter')

    def __init__(self, name, value=None, fixed=None, prior=None, ref=None, proposal=None, latex=None):
        """
        Initialize :class:`Parameter`.

        Parameters
        ----------
        name : tuple, list, string, ParamName, Parameter
            If :class:`Parameter`, update ``self`` attributes.
            Else, arguments to initialize parameter name, see :class:`ParamName`.

        value : float, default=False
            Default value for parameter.

        fixed : bool, default=None
            Whether parameter is fixed.
            If ``None``, defaults to ``True`` if ``prior`` or ``ref`` is not ``None``, else ``False``.

        prior : Prior, dict, default=None
            Prior distribution for parameter, arguments for :class:`Prior`.

        ref : Prior, dict, default=None
            Reference distribution for parameter, arguments for :class:`Prior`.
            This is supposed to represent the expected posterior for this parameter.
            If ``None``, defaults to ``prior``.

        proposal : float, default=None
            Proposal uncertainty for parameter.
            If ``None``, defaults to scale (or half of limiting range) of ``ref``.

        latex : string, default=None
            Latex for parameter.
        """
        if isinstance(name,Parameter):
            self.__dict__.update(name.__dict__)
            return
        self.name = ParamName(name)
        self.value = value
        self.prior = prior if isinstance(prior,Prior) else Prior(**(prior or {}))
        if value is None:
            if self.prior.is_proper():
                self.value = np.mean(self.prior.limits)
        if ref is not None:
            self.ref = ref if isinstance(ref,Prior) else Prior(**(ref or {}))
        else:
            self.ref = self.prior.copy()
        if value is None:
            if (ref is not None or prior is not None):
                if hasattr(self.ref,'loc'):
                    self.value = self.ref.loc
                elif self.ref.is_proper():
                    self.value = (self.ref.limits[1] - self.ref.limits[0])/2.
        self.latex = latex
        if fixed is None:
            fixed = prior is None and ref is None
        self.fixed = bool(fixed)
        self.proposal = proposal
        if proposal is None:
            if (ref is not None or prior is not None):
                if hasattr(self.ref,'scale'):
                    self.proposal = self.ref.scale
                elif self.ref.is_proper():
                    self.proposal = (self.ref.limits[1] - self.ref.limits[0])/2.

    def update(self, **kwargs):
        """Update parameter attributes with new arguments **kwargs**."""
        state = {key: getattr(self,key) for key in self._attrs}
        state.update(kwargs)
        self.__init__(**state)

    def add_suffix(self, suffix):
        """
        Add suffix to parameter:

        - update :attr:`name`
        - update :attr:`latex`
        """
        self.name.add_suffix(suffix)
        if self.latex is not None:
            match1 = re.match('(.*)_(.)$',self.latex)
            match2 = re.match('(.*)_{(.*)}$',self.latex)
            if match1 is not None:
                self.latex = '%s_{%s,\\mathrm{%s}}' % (match1.group(1),match1.group(2),suffix)
            elif match2 is not None:
                self.latex = '%s_{%s,\\mathrm{%s}}' % (match2.group(1),match2.group(2),suffix)
            else:
                self.latex = '%s_{\\mathrm{%s}}' % (self.latex,suffix)

    @property
    def varied(self):
        """Whether parameter is varied (i.e. not fixed)."""
        return (not self.fixed)

    def get_label(self):
        """If :attr:`latex` is specified (i.e. not ``None``), return :attr:`latex` surrounded by '$' signs, else ``None``."""
        if self.latex is not None:
            return '${}$'.format(self.latex)
        return self.name

    @property
    def limits(self):
        """Parameter limits."""
        return self.prior.limits

    def __getstate__(self):
        """Return this class state dictionary."""
        state = {}
        for key in self._attrs:
            state[key] = getattr(self,key)
            if hasattr(state[key],'__getstate__'):
                state[key] = state[key].__getstate__()
        return state

    def __setstate__(self, state):
        """Set this class state dictionary."""
        super(Parameter,self).__setstate__(state)
        self.name = ParamName.from_state(state['name'])
        for key in ['prior','ref']:
            setattr(self,key,Prior.from_state(state[key]))

    def __repr__(self):
        """Represent parameter as string (name and fixed or varied)."""
        return '{}({}, {})'.format(self.__class__.__name__,self.name,'fixed' if self.fixed else 'varied')

    def __str__(self):
        """Return parameter as string (name)."""
        return str(self.name)

    def __eq__(self, other):
        """Is ``self`` equal to ``other``, i.e. same type and attributes?"""
        return type(other) == type(self) and all(getattr(other,key) == getattr(self,key) for key in self._attrs)


class PriorError(Exception):

    """Exception raised when issue with prior."""


class Prior(BaseClass):
    """
    Class that describes a 1D prior distribution.

    Parameters
    ----------
    dist : string
        Distribution name.

    rv : scipy.stats.rv_continuous
        Random variate.

    attrs : dict
        Arguments used to initialize :attr:`rv`.
    """
    logger = logging.getLogger('Prior')

    def __init__(self, dist='uniform', limits=None, **kwargs):
        """
        Initialize :class:`Prior`.

        Parameters
        ----------
        dist : string
            Distribution name in :mod:`scipy.stats`

        limits : tuple, default=None
            Limits. See :meth:`set_limits`.

        kwargs : dict
            Arguments for :func:`scipy.stats.dist`, typically ``loc``, ``scale``
            (mean and standard deviation in case of a normal distribution ``'dist' == 'norm'``)
        """
        if isinstance(dist,Prior):
            self.__dict__.update(dist.__dict__)
            return

        self.set_limits(limits)
        self.dist = dist
        self.attrs = kwargs

        # improper prior
        if not self.is_proper():
            return

        if self.is_limited():
            dist = getattr(stats,self.dist if self.dist.startswith('trunc') or self.dist == 'uniform' else 'trunc{}'.format(self.dist))
            if self.dist == 'uniform':
                self.rv = dist(self.limits[0],self.limits[1]-self.limits[0])
            else:
                self.rv = dist(*self.limits,**kwargs)
        else:
            self.rv = getattr(stats,self.dist)(**kwargs)

    def set_limits(self, limits=None):
        r"""
        Set limits.

        Parameters
        ----------
        limits : tuple, default=None
            Tuple corresponding to lower, upper limits.
            ``None`` means :math:`-\infty` for lower bound and :math:`\infty` for upper bound.
            Defaults to :math:`-\infty,\infty`.
        """
        if not limits:
            limits = (-np.inf,np.inf)
        self.limits = list(limits)
        if self.limits[0] is None: self.limits[0] = -np.inf
        if self.limits[1] is None: self.limits[1] = np.inf
        self.limits = tuple(self.limits)
        if self.limits[1] <= self.limits[0]:
            raise PriorError('Prior range {} has min greater than max'.format(self.limits))
        if np.isinf(self.limits).any():
            return 1
        return 0

    def isin(self, x):
        """Whether ``x`` is within prior, i.e. within limits - strictly positive probability."""
        return self.limits[0] < x < self.limits[1]

    def __call__(self, x):
        """Return probability density at ``x``."""
        if not self.is_proper():
            return 1.*self.isin(x)
        return self.logpdf(x)

    def sample(self, size=None, random_state=None):
        """
        Draw ``size`` samples from prior. Possible only if prior is proper.

        Parameters
        ---------
        size : int, default=None
            Number of samples to draw.
            If ``None``, return one sample (float).

        random_state : int, numpy.random.Generator, numpy.random.RandomState, default=None
            If integer, a new :class:`numpy.random.RandomState` instance is used, seeded with ``random_state``.
            If ``random_state`` is a :class:`numpy.random.Generator` or :class:`numpy.random.RandomState` instance then that instance is used.
            If ``None``, the :class:`numpy.random.RandomState` singleton is used.

        Returns
        -------
        samples : float, array
            Samples drawn from prior.
        """
        if not self.is_proper():
            raise PriorError('Cannot sample from improper prior')
        return self.rv.rvs(size=size,random_state=random_state)

    def __str__(self):
        """Return string with distribution name, limits, and attributes (e.g. ``loc`` and ``scale``)."""
        base = self.dist
        if self.is_limited():
            base = '{}[{}, {}]'.format(self.dist,*self.limits)
        return '{}({})'.format(base,self.attrs)

    def __setstate__(self, state):
        """Set this class state dictionary."""
        self.__init__(state['dist'],state['limits'],**state['attrs'])

    def __getstate__(self):
        """Return this class state dictionary."""
        state = {}
        for key in ['dist','limits','attrs']:
            state[key] = getattr(self,key)
        return state

    def is_proper(self):
        """Whether distribution is proper, i.e. has finite integral."""
        return self.dist != 'uniform' or not np.isinf(self.limits).any()

    def is_limited(self):
        """Whether distribution has (at least one) finite limit."""
        return not np.isinf(self.limits).all()

    def __getattribute__(self, name):
        """Make :attr:`rv` attributes directly available in :class:`Prior`."""
        try:
            return object.__getattribute__(self,name)
        except AttributeError:
            attrs = object.__getattribute__(self,'attrs')
            if name in attrs:
                return attrs[name]
            rv = object.__getattribute__(self,'rv')
            return getattr(rv,name)

    def __eq__(self, other):
        """Is ``self`` equal to ``other``, i.e. same type and attributes?"""
        return type(other) == type(self) and all(getattr(other,key) == getattr(self,key) for key in ['dist','limits','attrs'])
