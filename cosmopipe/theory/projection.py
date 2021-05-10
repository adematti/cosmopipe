from cosmopipe.lib.theory import projection
from cosmopipe import section_names


class DataVectorProjection(object):

    def setup(self):
        data_vector = self.data_block[section_names.data,'data_vector']
        integration = self.options.get('integration',None)
        self.projection = projection.DataVectorProjection(xdata=data_vector,basemodel='xmu',integration=integration)

    def execute(self):
        y_callable = self.data_block[section_names.model,'y_callable']
        self.data_block[section_names.model,'y'] = self.projection(y_callable)

    def cleanup(self):
        pass
