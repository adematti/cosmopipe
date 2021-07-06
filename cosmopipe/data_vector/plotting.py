from pypescript import syntax

from cosmopipe import section_names
from cosmopipe.lib.data_vector import DataVector, DataPlotStyle
from cosmopipe.lib import syntax, utils


class DataVectorPlotting(object):

    def setup(self):
        self.covariance_load = self.options.get('covariance_load',None)
        if isinstance(self.covariance_load,bool) and self.covariance_load:
            self.covariance_load = 'covariance_matrix'
        if self.covariance_load:
            self.covariance_load = syntax.split_sections(self.covariance_load,default_section=section_names.covariance)
        self.data_load = self.options.get('data_load','data_vector')
        self.style = DataPlotStyle(**syntax.remove_keywords(self.options))

    def execute(self):
        self.style.mpicomm = self.mpicomm
        data = syntax.load_auto(self.data_load,data_block=self.data_block,default_section=section_names.data,loader=DataVector.load_auto)
        if self.covariance_load:
            self.style.plot(data,covariance=self.data_block[self.covariance_load])
        else:
            self.style.plot(data)

    def cleanup(self):
        pass


class CovarianceMatrixPlotting(object):

    def setup(self):
        self.covariance_load = syntax.split_sections(self.options.get('covariance_load','covariance_matrix'),default_section=section_names.covariance)
        self.style = self.options.get('style','corr')
        self.style = DataPlotStyle(self.style,**self.options)

    def execute(self):
        self.style.mpicomm = self.mpicomm
        covariance = self.data_block[self.covariance_load]
        self.style.plot(covariance)

    def cleanup(self):
        pass
