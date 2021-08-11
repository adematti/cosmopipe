"""Base class for linear transformations of the theory model."""

import numpy as np

from cosmopipe.lib.utils import BaseClass
from cosmopipe.lib.data_vector import ProjectionName, ProjectionNameCollection
from cosmopipe.lib.theory import ProjectionBasis


class BaseMatrix(BaseClass):
    """
    Base class to represent a linear transform of the theory model,
    from input projections :attr:`projsin` to output projections :attr:`projsout`.

    Attributes
    ----------
    basis : ProjectionBasis
        Projection basis the transform takes place in (e.g. Fourier space multipoles, wedges?)

    regularin : bool
        Should input array be regular, i.e. same length for each projection?

    regularout : bool
        Should output array be regular, i.e. same length for each projection?

    matrix : array
        2D array representing linear transform.

    projsin : ProjectionNameCollection
        Input projection names.

    projsout : ProjectionNameCollection
        Output projection names.
    """
    basis = ProjectionBasis()
    regularin = False
    regularout = False

    def __getstate__(self):
        """Return this class state dictionary."""
        state = {}
        for key in ['matrix']:
            state[key] = getattr(self,key)
        for key in ['projsin','projsout']:
            state[key] = ProjectionNameCollection(getattr(self,key)).__getstate__()
        return state

    def __setstate__(self, state):
        """Set this class state dictionary."""
        super(BaseMatrix,self).__setstate__(state)
        for key in ['projsin','projsout']:
            setattr(self,key,ProjectionNameCollection.__setstate__(getattr(self,key)))

    def compute(self, array):
        """Apply transform to input array."""
        return self.matrix.dot(array)

    def propose_out(self, projsin):
        """Propose input and output projection names given proposed input projection names ``projsin``."""
        projsout = ProjectionNameCollection(projsin).select(**self.basis.as_dict(drop_none=True))
        return projsout, projsout


class BaseRegularMatrix(BaseMatrix):
    """
    Base class to represent a linear transform of the theory model,
    with same array size for each input and output projections.

    Attributes
    ----------
    x : array
        x-coordinates, same for each input and output projections.
    """
    regularin = True
    regularout = True

    @property
    def xin(self):
        """Input x-coordinates."""
        return self.x

    @property
    def xout(self):
        """Output x-coordinates."""
        return self.x

    def __getstate__(self):
        """Return this class state dictionary."""
        state = super(BaseRegularMatrix,self).__getstate__()
        for key in ['x']:
            state[key] = getattr(self,key)
        return state


class ProjectionConversion(BaseRegularMatrix):
    """
    Class that handles conversion between different projection bases.

    Attributes
    ----------
    projmatrix : array
        Array of shape ``(len(self.projsout),len(self.projsin))``
        to convert input array from one basis to another (e.g. multipoles to wedges).
    """
    def __init__(self, x, projsin, projsout=None):
        """
        Initialize :class:`ProjectionConversion`.

        Parameters
        ----------
        x : array
            x-coordinates, same for each input and output projections.

        projsin : list, ProjectionNameCollection
            Input projections.

        projsout : list, ProjectionNameCollection, default=None
            Output projections. Defaults to ``projsin``.
        """
        self.projsin = ProjectionNameCollection(projsin)
        self.projsout = ProjectionNameCollection(projsout or projsin)
        modein = self.projsin.unique('mode')
        if len(modein) > 1:
            raise NotImplementedError('Cannot treat different input modes')
        modein = modein[0]
        self.x = np.asarray(x)
        matrix = []
        for projout in self.projsout:
            if modein == projout.mode:
                # crossing muwedges not considered here...
                #eq_ignore_none ensures we sum over wa_order
                line = [1.*projin.eq_ignore_none(projout) for projin in projsin]
                if modein == ProjectionName.MUWEDGE and np.max(line) != 1.:
                    raise NotImplementedError('Cannot interpolate between different mu-wedges')
            elif projout.mode == ProjectionName.MULTIPOLE:
                line = (2.*projout.proj + 1) * MultipoleToMuWedge([projout.proj],[proj.proj for proj in projsin]).weights.flatten()
            elif projout.mode == ProjectionName.MUWEDGE:
                line = MultipoleToMuWedge([proj.proj for proj in projsin],[projout.proj]).weights.flatten()
            matrix.append(line)
        self.projmatrix = np.array(matrix)

    @property
    def matrix(self):
        """
        Return 2D array of shape ``(len(self.projsout)*len(self.x),len(self.projsin)*len(self.x))``
        corresponding to :attr:`projmatrix`.
        """
        return np.bmat([[tmp*np.eye(self.x.size) for tmp in line] for line in self.projmatrix]).A

    @classmethod
    def propose_out(cls, projsin, baseout):
        """
        Propose output projection names given proposed input projection names ``projsin``.

        Parameters
        ----------
        projsin : list, ProjectionNameCollection
            Input projections.

        baseout : ProjectionBasis
            Projection basis for output.

        Returns
        -------
        toret : ProjectionNameCollection
            Proposed projection names.
        """
        projsin = ProjectionNameCollection(projsin)
        #print([proj.space for proj in projsin],baseout.space)
        if not all(proj.space == baseout.space or proj.space is None for proj in projsin):
            raise NotImplementedError('Space conversion not supported (yet)')
        modein = projsin.unique('mode')
        if len(modein) > 1:
            raise NotImplementedError('Cannot treat different input modes')
        modein = modein[0]
        if modein == baseout.mode or modein is None:
            return ProjectionNameCollection([proj.copy(mode=baseout.mode) for proj in projsin])
        groups = projsin.group_by(exclude=['proj']) # group by same attributes (e.g. wa_order) - except proj
        toret = []
        if modein == ProjectionName.MULTIPOLE and baseout.mode == ProjectionBasis.MUWEDGE:
            for projs in groups.values():
                ells = np.array([proj.proj for proj in projs])
                if np.all(ells % 2 == 0): # all even multipoles, propose wedges between 0 and 1
                    muwedges = np.linspace(0.,1.,len(ells)+1)
                else: # full (-1, 1) wedges
                    muwedges = np.linspace(-1.,1.,len(ells)+1)
                muwedges = zip(muwedges[:-1],muwedges[1:])
                toret += [proj.copy(mode=baseout.mode,proj=muwedge) for proj,muwedge in zip(projs,muwedges)]
            return ProjectionNameCollection(toret)
        if modein == ProjectionName.MUWEDGE and baseout.mode == ProjectionName.MULTIPOLE:
            for projs in groups.values():
                muwedges = [proj.proj for proj in projs]
                if np.all(np.array(muwedges) >= 0): # all positive wedges, propose even multipoles
                    ells = np.arange(0,2*len(muwedges)+1,2)
                else: # all (even and odd) multipoles
                    ells = np.arange(0,len(muwedges)+1,1)
                toret += [proj.copy(mode=baseout.mode,proj=ell) for proj,ell in zip(projs,ells)]
            return ProjectionNameCollection(toret)
        raise NotImplementedError('Unknown conversion {} -> {}'.format(modein,baseout))
