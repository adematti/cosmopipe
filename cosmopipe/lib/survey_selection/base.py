import numpy as np

from cosmopipe.lib.utils import BaseClass
from cosmopipe.lib.data_vector import ProjectionName, ProjectionNameCollection
from cosmopipe.lib.theory import ProjectionBase


class BaseMatrix(BaseClass):

    base = ProjectionBase()
    regularin = False
    regularout = False

    def __getstate__(self):
        state = {}
        for key in ['matrix']:
            state[key] = getattr(self,key)
        for key in ['projsin','projsout']:
            state[key] = ProjectionNameCollection(getattr(self,key)).__getstate__()
        return state

    def __setstate__(self, state):
        super(BaseMatrix,self).__setstate__(state)
        for key in ['projsin','projsout']:
            setattr(self,key,ProjectionNameCollection.__setstate__(getattr(self,key)))

    def compute(self, array):
        return self.matrix.dot(array)

    def propose_out(self, projsin):
        projsout = ProjectionNameCollection(projsin).select(**self.base.as_dict(drop_none=True))
        return projsout, projsout


class BaseRegularMatrix(BaseMatrix):

    regularin = True
    regularout = True

    @property
    def xin(self):
        return self.x

    @property
    def xout(self):
        return self.x

    def __getstate__(self):
        state = super(BaseRegularMatrix,self).__getstate__()
        for key in ['x']:
            state[key] = getattr(self,key)
        return state


class ProjectionConversion(BaseRegularMatrix):

    def __init__(self, x, projsin, projsout=None):
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
        self.matrix = np.bmat([[tmp*np.eye(self.x.size) for tmp in line] for line in matrix]).A

    @classmethod
    def propose_out(cls, projsin, baseout=None):
        projsin = ProjectionNameCollection(projsin)
        if baseout is None:
            return projsin.copy()
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
        if modein == ProjectionName.MULTIPOLE and baseout.mode == ProjectionBase.MUWEDGE:
            for projs in groups.values():
                ells = np.array([proj.proj for proj in projs])
                if np.all(ells % 2 == 0):
                    muwedges = np.linspace(0.,1.,len(ells)+1)
                else:
                    muwedges = np.linspace(-1.,1.,len(ells)+1)
                muwedges = zip(muwedges[:-1],muwedges[1:])
                toret += [proj.copy(mode=baseout.mode,proj=muwedge) for proj,muwedge in zip(projs,muwedges)]
            return toret
        if modein == ProjectionName.MUWEDGE and baseout.mode == ProjectionName.MULTIPOLE:
            for projs in groups.values():
                muwedges = [proj.proj for proj in projs]
                if np.all(np.array(muwedges) >= 0):
                    ells = np.arange(0,2*len(muwedges)+1,2)
                else:
                    ells = np.arange(0,len(muwedges)+1,1)
                toret += [proj.copy(mode=baseout.mode,proj=ell) for proj,ell in zip(projs,ells)]
            return toret
        raise NotImplementedError('Unknown conversion {} -> {}'.format(modein,baseout))
