import numpy as np

from cosmopipe.lib.utils import BaseClass
from cosmopipe.lib.data_vector import DataVector, ProjectionName, ProjectionNameCollection
from cosmopipe.lib.theory import ProjectionBase, ProjectionBaseCollection, ModelEvaluation

from .base import ProjectionConversion, BaseRegularMatrix
from .binning import BaseBinning


class ModelProjectionCollection(BaseClass):

    def __init__(self, data, projs=None, model_base=None, integration=None):

        #self.model_bases = ProjectionBaseCollection(model_base)
        self.data = data
        self.projs = ProjectionNameCollection(projs) if projs is not None else data.get_projs()

        self.groups = self.projs.group_by(exclude=['mode','proj','wa_order'])
        self.projection_mapping = [None]*len(self.projs)
        nprojs = 0
        self.model_projections = []
        for label,projs in self.groups.items():
            indices = [self.projs.index(proj) for proj in projs]
            for ii,jj in enumerate(indices): self.projection_mapping[jj] = nprojs + ii
            nprojs += len(projs)
            modelproj = ModelProjection(self.data,projs=projs,model_base=model_base,integration=integration)
            self.model_projections.append(modelproj)

    def copy(self, *args, **kwargs):
        new = self.__copy__()
        new.model_projections = [modelproj.copy(*args,**kwargs) for modelproj in self.model_projections]
        return new

    def append(self, *args, **kwargs):
        modelproj = ModelProjection(*args,**kwargs)
        add_modelproj = False
        for proj in modelproj.projs:
            add_modelproj = proj not in self.projs
            if add_modelproj:
                self.projs.set(proj)
                self.projection_mapping.append(len(self.projs))
        if add_modelproj:
            self.data += modelproj.data
            self.model_projections.append(modelproj)

    def setup(self, data=None, projs=None):

        if data is not None:
            self.data = data

        if data is not None or projs is not None:
            self_group_labels = list(self.groups.keys())
            self_model_projections = self.model_projections
            self.projs = ProjectionNameCollection(projs) if projs is not None else data.get_projs()
            self.groups = self.projs.group_by(exclude=['mode','proj','wa_order'])
            self.projection_mapping = [None]*len(self.projs)
            nprojs = 0
            self.model_projections = []
            for label,projs in self.groups.items():
                indices = [projs.index(proj) for proj in projs]
                for ii,jj in enumerate(indices): self.projection_mapping[jj] = nprojs + ii
                self.model_projections.append(self_model_projections[self_group_labels.index(label)])

        for model_projection,projs in zip(self.model_projections,self.groups.values()):
            model_projection.setup(data=data,projs=projs)

    @classmethod
    def concatenate(cls, *others):
        new = others[0].copy()
        for other in others[1:]:
            for item in other.model_projections:
                new.append(item)
        return new

    def extend(self, other):
        new = self.concatenate(self,other)
        self.__dict__.update(new.__dict__)

    def __radd__(self, other):
        if other in [[],0,None]:
            return self.copy()
        return self.concatenate(self,other)

    def __iadd__(self, other):
        self.extend(other)
        return self

    def __add__(self, other):
        return self.concatenate(self,other)

    def __call__(self, models, concatenate=True, **kwargs):
        tmp = []
        for model_projection in self.model_projections:
            tmp += model_projection(models,concatenate=False, **kwargs)
        toret = [tmp[ii] for ii in self.projection_mapping]
        if concatenate:
            return np.concatenate(toret)
        return toret

    def to_data_vector(self, models, **kwargs):
        y = self(models,concatenate=True,**kwargs)
        data = self.data.deepcopy()
        data.set_y(y,proj=self.projs)
        return data


class ModelProjection(BaseClass):

    def __init__(self, data, projs=None, model_base=None, integration=None):

        if isinstance(data,self.__class__):
            self.__dict__.update(data.__dict__)
            return

        self.single_model = not isinstance(model_base,(list,ProjectionBaseCollection))
        self.model_bases = ProjectionBaseCollection(model_base)

        self.data = data
        self.projs = ProjectionNameCollection(projs) if projs is not None else data.get_projs()
        self.operations = []
        self.integration_options = integration

    def copy(self, copy_operations=True):
        new = self.__copy__()
        if copy_operations:
            new.operations = [operation.copy() for operation in self.operations]
        else:
            new.operations = self.operations.copy()
        return new

    def insert(self, i, operation):
        self.operations.insert(i,operation)

    def append(self, operation):
        self.operations.append(operation)

    def setup(self, data=None, projs=None):
        if data is not None:
            self.data = data
        if projs is not None:
            self.projs = ProjectionNameCollection(projs)
        operations = self.operations.copy()
        if not operations or not isinstance(operations[-1],BaseBinning):
            binning = BaseBinning()
            operations.append(binning)

        # this is binning operation
        operations[-1].setup(data=self.data,projs=self.projs)

        operation = operations[0]
        projsout = ModelEvaluation.propose_out(self.model_bases)
        projsin,projsout = operation.propose_out(projsout)
        projsins = [projsin]
        projsouts = [projsout]
        for operation in operations[1:-1]:
            projsout = ProjectionConversion.propose_out(projsout,baseout=operation.base)
            projsin,projsout = operation.propose_out(projsout) # here we can change projsin, projsout
            #print('SETUP',operation.__class__)
            #print('IN',projsin)
            #print('OUT',projsout)
            #print('')
            projsins.append(projsin)
            projsouts.append(projsout)

        self.matrix = operations[-1].matrix # binning
        for projsin,projsout,operation,next in zip(projsins[::-1],projsouts[::-1],operations[-2::-1],operations[:0:-1]):
            if operation.regularout:
                if not next.regularin:
                    raise ValueError('An interpolation must be performed between {} and {}'.format(operation.__class__.__name__,next.__class__.__name__))
                conversion = ProjectionConversion(next.xin,projsout=next.projsin,projsin=projsout)
                x = conversion.xin
                projsout = conversion.projsin
                self.matrix = self.matrix @ conversion.matrix
            else:
                x = next.xin
                projsout = next.projsin
            operation.setup(x,projsin=projsin,projsout=projsout) # setup can reduce projsin
            self.matrix = self.matrix @ operation.matrix
        if np.allclose(self.matrix,np.eye(*self.matrix.shape,dtype=self.matrix.dtype)):
            self.matrix = None

        operation = operations[0]
        data = DataVector(x=operation.xin,proj=operation.projsin)

        self.evaluation = ModelEvaluation(data=data,model_base=self.model_bases[0] if self.single_model else self.model_bases,integration=self.integration_options)

        xout = operations[-1].xout
        cumsizes = np.cumsum([0] + [len(x) for x in xout])
        self.slices = [slice(start,stop) for start,stop in zip(cumsizes[:-1],cumsizes[1:])]

    def __call__(self, models, concatenate=True, **kwargs):
        toret = self.evaluation(models,concatenate=True,remove_shotnoise=True,**kwargs)
        if self.matrix is not None:
            toret = self.matrix.dot(toret)
        if not concatenate:
            return [toret[sl] for sl in self.slices]
        return toret

    def to_data_vector(self, models, **kwargs):
        y = self(models,concatenate=True,**kwargs)
        data = self.data.deepcopy()
        #print(y.size)
        data.set_y(y,proj=self.projs)
        return data
