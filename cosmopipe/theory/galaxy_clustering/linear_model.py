import numpy as np

from cosmopipe.lib import theory
from cosmopipe.lib.theory.base import ProjectionBase, ModelCollection

from cosmopipe import section_names


class LinearModel(object):

    def setup(self):
        zeff = self.data_block[section_names.survey_geometry,'zeff']
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
        self.data_block[section_names.model,'collection'] = self.data_block.get(section_names.model,'collection',[]) + ModelCollection([self.model])
        self.x = pklin.k

    def execute(self):
        sigmav = self.data_block.get(section_names.galaxy_bias,'sigmav',0.)
        b1 = self.data_block.get(section_names.galaxy_bias,'b1',1.)
        shotnoise = (1. + self.data_block.get(section_names.galaxy_bias,'As',0.))*self.data_shotnoise
        fsig = self.data_block.get(section_names.galaxy_rsd,'fsig',self.growth_rate*self.sigma8)

        def model(k, mu, grid=True):
            return self.eval(k,mu,b1=b1,sigmav=sigmav,shotnoise=shotnoise,f=fsig/self.sigma8,grid=grid)

        self.model.eval = model
        self.data_block[section_names.model,'collection'] = self.data_block.get(section_names.model,'collection',[]) + ModelCollection([self.model])

    def cleanup(self):
        pass
