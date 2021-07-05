import numpy as np

from cosmopipe import section_names
from cosmopipe.lib.data_vector import mock_data_vector

from .data_vector import get_kwview


class MockDataVector(object):

    def setup(self):
        self.seed = self.options.get_int('seed',None)
        self.mean = self.options.get_bool('mean',False)
        self.save = self.options.get_string('save',False)
        covariance = self.data_block[section_names.covariance,'covariance_matrix']
        #seed = self.data_block.get(section_names.data,'seed',self.seed)
        data = mock_data_vector.MockDataVector(covariance,seed=self.seed,mean=self.mean)
        if self.save: data.save_auto(self.save)
        data = data.view(**get_kwview(data,xlim=self.options.get('xlim',None)))
        self.data_block[section_names.data,'data_vector'] = data
        self.data_block[section_names.data,'y'] = data.get_y()

    def execute(self):
        pass

    def cleanup(self):
        pass