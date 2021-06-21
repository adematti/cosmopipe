from cosmopipe.lib.theory import projection
from cosmopipe.lib import syntax, utils
from cosmopipe import section_names


class DataVectorProjection(object):

    def setup(self):
        integration = self.options.get('integration',None)
        projs = self.options.get('projs',None)
        xmodels = self.options.get('x',None)
        model_base = self.data_block[section_names.model,'y_base']
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
            self.projection = projection.DataVectorProjection(x=x,projs=projs,model_base=model_base,integration=integration)
        else:
            self.projection = projection.DataVectorProjection(x=self.data_block[section_names.data,'data_vector'],projs=projs,model_base=model_base,integration=integration)

    def execute(self):
        y_callable = self.data_block[section_names.model,'y_callable']
        self.data_block[section_names.model,'y'] = self.projection(y_callable)
        self.data_block[section_names.model,'projection'] = self.projection

    def cleanup(self):
        pass
