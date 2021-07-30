import numpy as np

from cosmopipe import section_names
from cosmopipe.lib import data_vector

from .data_vector import get_data_from_options, get_kwview


class MockCovarianceMatrix(object):

    def setup(self):
        self.data_load = self.options.get_list('data_load')
        list_data = []
        for data_load in self.data_load:
            data = get_data_from_options(self.options,data_load=data_load,data_block=self.data_block)
            list_data.append(data)

        cov = data_vector.MockCovarianceMatrix.from_data(list_data)
        xlim = self.options.get('xlim',None)
        if xlim is not None:
            kwview = get_kwview(list_data[0],xlim=xlim)
            cov = cov.view(**kwview)

        save = self.options.get('save',None)
        if save: cov.save_auto(self.save)

        self.data_block[section_names.covariance,'covariance_matrix'] = cov
        self.data_block[section_names.covariance,'cov'] = cov.get_cov()
        self.data_block[section_names.covariance,'invcov'] = cov.get_invcov()
        self.data_block[section_names.covariance,'nobs'] = cov.attrs.get('nobs',None)

    def execute(self):
        pass

    def cleanup(self):
        pass
