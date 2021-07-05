from cosmopipe.lib.theory import projection
from cosmopipe.lib import syntax, utils
from cosmopipe import section_names


class ModelProjection(object):

    def setup(self):
        integration = self.options.get('integration',None)
        projs = self.options.get('projs',None)
        xmodels = self.options.get('x',None)
        model_bases = self.data_block[section_names.model,'collection'].bases
        data_vector = self.data_block.get(section_names.data,'data_vector',None)
        if xmodels is not None:
            x = []
            if isinstance(xmodels,dict) or np.ndim(xmodels[0]) == 0:
                xmodels = [xmodels]*len(projs)
            for xmodel,proj in zip(xmodels,projs):
                kwargs = xmodel.copy()
                if isinstance(xmodel,dict):
                    if xmodel.get('min',None) is None:
                        kwargs['min'] = data_vector.get_x(proj=proj).min()
                    if xmodel.get('max',None) is None:
                        kwargs['max'] = data_vector.get_x(proj=proj).max()
                    x.append(utils.customspace(**kwargs))
                else:
                    x.append(xmodel)
            data = DataVector(x=x,proj=projs)
        else:
            data = data_vector
        self.projection = projection.ModelCollectionProjection(data=data,projs=projs,model_bases=model_bases,integration=integration)

    def execute(self):
        collection = self.data_block[section_names.model,'collection']
        self.data_block[section_names.model,'y'] = self.projection(collection)
        self.data_block[section_names.model,'projection'] = self.projection

    def cleanup(self):
        pass
