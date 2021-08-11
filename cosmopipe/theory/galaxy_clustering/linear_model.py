import numpy as np

from cosmopipe.lib import theory
from cosmopipe.lib.theory.base import ProjectionBasis, ModelCollection
from cosmopipe import section_names

from .module import PTModule


class LinearModel(PTModule):

    def setup(self):
        self.set_parameters()
        self.set_primordial()
        model_attrs = self.options.get_dict('model_attrs',{})
        self.model = theory.LinearModel(klin=self.pklin.k,pklin=self.pklin,FoG=self.options.get('FoG','gaussian'))
        self.model.basis.set(**model_attrs)
        self.eval = self.model.eval
        self.data_shotnoise = self.options.get('data_shotnoise',None)
        try:
            self.data_shotnoise = 1. * self.data_shotnoise
        except TypeError:
            self.data_shotnoise = self.data_block[section_names.data,'data_vector'].get(self.data_shotnoise,permissive=True)[0].attrs['shotnoise']
        self.model.basis.set(shotnoise=self.data_shotnoise,**self.options.get('model_attrs',{}))
        self.model.eval = None # just to make sure it is not called without execute
        model_collection = self.data_block.get(section_names.model,'collection',[])
        model_collection += ModelCollection([self.model])
        self.data_block[section_names.model,'collection'] = model_collection
        self.x = self.pklin.k

    def execute(self):
        if self.set_primordial():
            self.model.pk_linear = self.pklin

        fsig = self.data_block[section_names.galaxy_rsd,'fsig']
        sigmav = self.data_block.get(section_names.galaxy_bias,'sigmav',0.)
        #print(self._datablock_mapping)
        b1 = self.data_block.get(section_names.galaxy_bias,'b1',1.)
        shotnoise = (1. + self.data_block.get(section_names.galaxy_bias,'As',0.))*self.data_shotnoise

        def model(k, mu, grid=True):
            return self.eval(k,mu,b1=b1,sigmav=sigmav,shotnoise=shotnoise,f=fsig/self.sigma,grid=grid)

        self.model.eval = model

    def cleanup(self):
        pass
