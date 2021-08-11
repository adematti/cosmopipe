"""Definition of projection."""

import re

import numpy as np

from cosmopipe.lib.utils import BaseNameSpace, BaseOrderedCollection


class ProjectionName(BaseNameSpace):
    r"""
    Class describing a projection.
    All attributes default to ``None`` (not specified or not relevant).

    Attributes
    ----------
    name : string
        Name of projection.

    fields : tuple
        Tracer field(s).

    space : string
        Projection space, e.g. power spectrum ('power')? Correlation function ('correlation')?

    mode : string
        Projection mode, e.g. 'multipole'? 'muwedge'?

    proj : tuple, int
        Projection number or identifier (e.g. order of Legendre polynomial, lower and upper limit of :math:`\mu`-wedge)

    wa_order : int
        Wide-angle order.
    """
    MULTIPOLE = 'multipole'
    MUWEDGE = 'muwedge'
    MUBIN = 'mubin'
    PIWEDGE = 'piwedge'
    CORRELATION = 'correlation'
    POWER = 'power'
    _mode_shorts = {MULTIPOLE:'ell',MUWEDGE:'mu',MUBIN:'mubin',PIWEDGE:'pi',None:'None'}
    _space_shorts = {POWER:'power',CORRELATION:'corr'}
    _latex = {MULTIPOLE:'\ell',MUWEDGE:'\mu',MUBIN:'\mu',PIWEDGE:'\pi'}
    _attrs = ['name','fields','space','mode','proj','wa_order']
    sep = '_'

    def __init__(self, *args, **kwargs):
        """
        Initialize :class:`ProjectionName`.

        Example
        -------
        ``ProjectionName('ell_2') == Projection(mode='multipole',proj=2)``
        ``ProjectionName('multipole',2) == Projection(mode='multipole',proj=2)``
        ``ProjectionName(('multipole',2)) == Projection(mode='multipole',proj=2)``

        Parameters
        ----------
        args : tuple, list, string, dict, ProjectionName
            Can be:

            - a tuple or list of (mode, proj), (space, mode, proj) or (name, space, mode, proj)
            - string of mode_proj, space_mode_proj, or name_space_mode_proj
            - a dictionary of attribute values
            - a :class:`ProjectionName` instance, which is copied.

        kwargs : dict
            Dictionary of attribute values.
            If ``args`` is dictionary, is updated by ``kwargs``.
        """
        for name in self._attrs:
            setattr(self,name,None)
        if not len(args):
            pass
        elif len(args) > 1:
            if len(args) == 2:
                self.mode,self.proj = args
            elif len(args) == 3:
                self.space,self.mode,self.proj = args
            else:
                self.name,self.space,self.mode,self.proj = args
        elif isinstance(args[0],self.__class__):
            self.__dict__.update(args[0].__dict__)
        elif isinstance(args[0],dict):
            kwargs = {**args[0],**kwargs}
        elif isinstance(args[0],(list,tuple)):
            self.__init__(*args[0],**kwargs)
        elif isinstance(args[0],str):
            args = args[0].split(self.sep)
            self.name, self.space, self.mode = None, None, None
            for name,short in self._space_shorts.items():
                if args[0] == short:
                    self.space = name
                    args = args[1:]
                    break
                if args[1] == short:
                    self.name = args[0]
                    self.space = name
                    args = args[2:]
                    break
            for name,short in self._mode_shorts.items():
                if args[0] == short:
                    self.mode = name
                    args = args[1:]
                    break
                if self.name is None and args[1] == short:
                    self.name = args[0]
                    self.mode = name
                    args = args[2:]
                    break
            self.proj = tuple(eval(t,{},{}) for t in args)
            if len(args) == 1:
                self.proj = self.proj[0]
            if self.mode is None:
                raise ValueError('Cannot read projection {}'.format(args))
        self.set(**kwargs)

    def set(self, **kwargs):
        """Set projection attributes in ``kwargs``."""
        for name,value in kwargs.items():
            setattr(self,name,value)
        if np.ndim(self.proj):
            self.proj = tuple(self.proj)

    @property
    def latex(self):
        r"""Return *latex* (e.g., for the quadrupole, :math:`\ell = 2`)."""
        base = self._latex[self.mode]
        isscalar = np.ndim(self.proj) == 0
        proj = (self.proj,) if isscalar else self.proj
        label = ','.join(['{}'.format(p) if self.mode == self.MULTIPOLE else '{:.2f}'.format(p) for p in proj if p is not None])
        if not isscalar:
            label = '({})'.format(label)
        return '{} = {}'.format(base,label)

    def get_projlabel(self):
        """If :attr:`mode` is specified (i.e. not ``None``), return :attr:`latex` surrounded by '$' signs, else ``None``."""
        if self.mode is None:
            return None
        return '${}$'.format(self.latex)

    def get_xlabel(self):
        """Return x-coordinate label."""
        if self.space == self.POWER:
            return '$k$ [$h \ \\mathrm{Mpc}^{-1}$]'
        if self.space == self.CORRELATION:
            return '$s$ [$\\mathrm{Mpc} / h$]'

    def get_ylabel(self):
        """Return y-coordinate label."""
        if self.space == self.POWER:
            return '$P(k)$ [$(\\mathrm{Mpc} \ h)^{-1})^{3}$]'
        if self.space == self.CORRELATION:
            return '$\\xi(s)$'

    def __gt__(self, other):
        # Used for sorting
        return np.mean(self.proj) > np.mean(other.proj)

    def __lt__(self, other):
        # Used for sorting
        return np.mean(self.proj) < np.mean(other.proj)


class ProjectionNameCollection(BaseOrderedCollection):
    """
    Class describing a collection of projections.

    Note
    ----
    When adding a projection equal to another already in the collection, the latter will be replaced by the former.
    Insertion order is conserved.
    """
    _cast = lambda x: x if isinstance(x,ProjectionName) else ProjectionName(x)

    def index(self, proj, ignore_none=False):
        """
        Return index of :class:`ProjectionName` ``proj``.

        Parameters
        ----------
        proj : string, tuple, dict
            Arguments to initialize :class:`ProjectionName`.

        ignore_none : bool, default=False
            When comparing :class:`ProjectionName` instances,
            ignore ``None`` (unspecified) attributes.

        Returns
        -------
        index : int, list
            If ``ignore_none`` is ``True``, return list of all matches.
            Else return index of ``proj``.
        """
        proj = self.__class__._cast(proj)
        if ignore_none:
            return [iproj_ for iproj_,proj_ in enumerate(self.data) if proj.eq_ignore_none(proj_)]
        if proj not in self.data:
            raise KeyError('Projection {} not found among {}'.format(proj,self.data))
        return self.data.index(proj)

    def get(self, proj, ignore_none=True):
        """
        Return ``proj`` instance in ``self`` corresponding to :class:`ProjectionName` ``proj``.

        Parameters
        ----------
        proj : string, tuple, dict
            Arguments to initialize :class:`ProjectionName`.

        ignore_none : bool, default=False
            When comparing :class:`ProjectionName` instances,
            ignore ``None`` (unspecified) attributes.

        Returns
        -------
        proj : ProjectionName, list
            If ``ignore_none`` is ``True``, return list of all matching projections.
            Else return :class:`ProjectionName` instance matching ``proj``.
        """
        if ignore_none:
            return [self.data[ii] for ii in self.index(proj,ignore_none=ignore_none)]
        return self.data[self.data.index(proj)]

    def group_by(self, include=None, exclude=None):
        """
        Group :class:`ProjectionName` by similar attributes.

        Example
        -------
        ``collection.group_by(include=['space'])`` will group projections by space.
        ``collection.group_by(excluding=['mode'])`` will group projections ignoring differences in ``mode``.

        Parameters
        ----------
        include : list, default=None
            List of :class:`ProjectionName` attributes to form a group.

        exclude : list, default=None
            List of :class:`ProjectionName` attributes to ignore when forming a group.

        Returns
        -------
        toret : dict
            Dictionary of ``proj: collection``,
            with ``proj`` the :class:`ProjectionName` instance with attributes common
            to all projections in :class:`ProjectionNameCollection` ``collection``.
        """
        if not len(self):
            return {}
        include = include or []
        exclude = exclude or []
        exclude = exclude + [key for key in ProjectionName._attrs if key not in include]
        exclude = {key:None for key in exclude}
        toret = {}
        for proj in self.data:
            base = proj.copy(**exclude)
            if base not in toret:
                toret[base] = ProjectionNameCollection()
            toret[base].extend(proj)
        return toret
