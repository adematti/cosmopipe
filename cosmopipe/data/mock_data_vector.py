import numpy as np

from cosmopipe import section_names
from cosmopipe.lib.data import mock_data_vector

from .data_vector import get_kwview


class MockDataVector(object):

    def setup(self):
        self.seed = self.options.get('seed',None)
        self.mean = self.options.get('mean',False)
        self.save = self.options.get('save',False)
        self.save_txt = self.options.get('save_txt',False)

    def execute(self):
        covariance = self.data_block[section_names.covariance,'covariance_matrix']
        seed = self.data_block.get(section_names.data,'seed',self.seed)
        data = mock_data_vector.MockDataVector(covariance,seed=seed,mean=self.mean)
        if self.save: data.save(self.save)
        if self.save_txt: data.save_txt(self.save_txt,ignore_json_errors=True)
        data = data.view(**get_kwview(data,self.options))
        self.data_block[section_names.data,'xlims'] = np.array(data.kwview['xlim'])
        self.data_block[section_names.data,'projs'] = np.array(data.kwview['proj'])
        self.data_block[section_names.data,'data_vector'] = data
        self.data_block[section_names.data,'x'] = data.get_x()
        self.data_block[section_names.data,'y'] = data.get_y()
        self.data_block[section_names.data,'shotnoise'] = data.attrs.get('shotnoise',0.)

    def cleanup(self):
        pass
