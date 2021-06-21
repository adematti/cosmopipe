import numpy as np

from cosmopipe import section_names
from cosmopipe.lib.data import MockCovarianceMatrix

from .data_vector import get_data_from_options, get_kwview


class MockDataVector(object):

    def setup(self):
        self.data_load = self.options.get_list('data_load')
        list_data = []
        for data_load in self.data_load:
            data = get_data_from_options({**self.options,'data_load':data_load},data_block=self.data_block)
            list_data.append(data)
        proj,xlim = get_kwview(list_data[0],xlim=self.options.get('xlim',None))

        cov = MockCovarianceMatrix.from_data(list_data)
        self.data_block[section_names.covariance,'covariance_matrix'] = cov
        self.data_block[section_names.covariance,'cov'] = cov.get_cov()
        self.data_block[section_names.covariance,'invcov'] = cov.get_invcov()
        self.data_block[section_names.covariance,'nobs'] = cov.attrs.get('nobs',None)

    def execute(self):
        pass

    def cleanup(self):
        pass
