from pypescript import syntax

from cosmopipe import section_names
from cosmopipe.lib.data import DataPlotStyle
from cosmopipe.lib import utils


class DataVectorPlotting(object):

    def setup(self):
        self.covariance_key = syntax.split_sections(self.options.get('covariance_key','covariance_matrix'),default_section=section_names.covariance)
        data_keys = self.options.get('data_keys',None)
        if data_keys is None:
            data_keys = ['data_vector']
        if not isinstance(data_keys,list):
            data_keys = eval(data_keys)
        self.data_keys = []
        for key in data_keys:
            key = syntax.split_sections(key,default_section=section_names.data)
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
        self.covariance_key = syntax.split_sections(self.options.get('covariance_key','covariance_matrix'),default_section=section_names.covariance)
        self.style = DataPlotStyle(self.style,**self.options)

    def execute(self):
        covariance = self.data_block[self.covariance_key]
        self.style.plot(covariance)

    def cleanup(self):
        pass
