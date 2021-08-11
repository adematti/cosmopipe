import numpy as np

from cosmopipe import section_names
from cosmopipe.lib.data_vector import mock_data_vector
from cosmopipe.lib import syntax

from .data_vector import get_kwview


class MockDataVector(object):

    def setup(self):
        seed = self.options.get_int('seed',None)
        mean = self.options.get_bool('mean',False)
        save = self.options.get('save',False)
        covariance = self.data_block[section_names.covariance,'covariance_matrix']
        y = None
        mean_load = self.options.get('mean_load',False)
        if isinstance(mean_load,bool) and mean_load:
            mean_load = 'y'
        if mean_load:
            mean_load = syntax.split_sections(mean_load,default_section=section_names.model)
            y = self.data_block[mean_load]
        #seed = self.data_block.get(section_names.data,'seed',seed)
        data = mock_data_vector.MockDataVector(covariance,seed=seed,y=y,mean=mean)
        if save: data.save_auto(save)
        kwview = get_kwview(data,xlim=self.options.get('xlim',None))
        data = data.view(**kwview)
        self.data_block[section_names.data,'data_vector'] = data
        self.data_block[section_names.data,'y'] = data.get_y()

    def execute(self):
        pass

    def cleanup(self):
        pass
