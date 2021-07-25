import numpy as np

from cosmopipe.lib import theory
from cosmopipe.lib.theory.base import ProjectionBase, ModelCollection
from cosmopipe.lib.modules import ParameterizedModule

from cosmopipe import section_names


class LinearModel(ParameterizedModule):

    def setup(self):
        self.set_param_block()
        zeff = self.data_block[section_names.survey_selection,'zeff']
        pklin = self.data_block[section_names.primordial_perturbations,'pk_callable'].to_1d(z=zeff)
        self.sigma8 = pklin.sigma8()
        fo = self.data_block[section_names.primordial_cosmology,'cosmo'].get_fourier()
        self.growth_rate = fo.sigma8_z(zeff,of='theta_cb')/fo.sigma8_z(zeff,of='delta_cb')
        self.model = theory.LinearModel(klin=pklin.k,pklin=pklin,FoG=self.options.get('FoG','gaussian'))
        self.eval = self.model.eval
        self.data_shotnoise = self.options.get('data_shotnoise',None)
        try:
            self.data_shotnoise = 1. * self.data_shotnoise
        except TypeError:
            self.data_shotnoise = self.data_block[section_names.data,'data_vector'].get(self.data_shotnoise,permissive=True)[0].attrs['shotnoise']
        self.model.base.set(shotnoise=self.data_shotnoise,**self.options.get('model_attrs',{}))
        self.model.eval = None # just to make sure it is not called without execute
        model_collection = self.data_block.get(section_names.model,'collection',[])
        model_collection += ModelCollection([self.model])
        self.data_block[section_names.model,'collection'] = model_collection
        self.x = pklin.k

    def execute(self):
        sigmav = self.data_block.get(section_names.galaxy_bias,'sigmav',0.)
        #print(self._datablock_mapping)
        b1 = self.data_block.get(section_names.galaxy_bias,'b1',1.)
        shotnoise = (1. + self.data_block.get(section_names.galaxy_bias,'As',0.))*self.data_shotnoise
        fsig = self.data_block.get(section_names.galaxy_rsd,'fsig',self.growth_rate*self.sigma8)

        def model(k, mu, grid=True):
            return self.eval(k,mu,b1=b1,sigmav=sigmav,shotnoise=shotnoise,f=fsig/self.sigma8,grid=grid)

        self.model.eval = model

    def cleanup(self):
        pass
