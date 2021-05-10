from cosmopipe import section_names
from cosmopipe.lib.data import DataPlotStyle
from cosmopipe.lib import utils


class DataVectorPlotting(object):

    def setup(self):
        self.covariance_key = utils.split_section_name(self.options.get('covariance_key',None))
        if self.covariance_key is not None and len(self.covariance_key) == 1:
            self.covariance_key = (section_names.covariance,) + self.covariance_key
        data_keys = self.options.get('data_keys',None)
        if data_keys is None:
            data_keys = ['data_vector']
        if not isinstance(data_keys,list):
            data_keys = eval(data_keys)
        self.data_keys = []
        for key in data_keys:
            key = utils.split_section_name(key)
            if len(key) == 1:
                key = (section_names.data,) + key
            self.data_keys.append(key)
        self.style = DataPlotStyle(**self.options)

    def execute(self):
        data = []
        for key in self.data_keys:
            data.append(self.data_block[key])
        if self.covariance_key is not None:
            covariance = self.data_block[self.covariance_key]
            self.style.plot(data,covariance=covariance)
        else:
            self.style.plot(data)

    def cleanup(self):
        pass


class CovarianceMatrixPlotting(object):

    def setup(self):
        self.style = self.options['style']
        self.covariance_key = utils.split_section_name(self.options.get('covariance_key','covariance_matrix'))
        if len(self.covariance_key) == 1:
            self.covariance_key = (section_names.covariance,) + self.covariance_key
        self.style = DataPlotStyle(self.style,**self.options)

    def execute(self):
        covariance = self.data_block[self.covariance_key]
        self.style.plot(covariance)

    def cleanup(self):
        pass
