import logging

from pypescript import BaseModule
from nbodykit.lab import FFTPower

from cosmopipe.lib.catalog import Catalog, utils


class BoxPowerSpectrum(BaseModule):

    logger = logging.getLogger('BoxPowerSpectrum')

    def setup(self):
        self.mesh_options = {'Nmesh':512,'BoxSize':None,'resampler':'tsc','interlaced':True}
        for name,value in self.mesh_options.items():
            self.mesh_options[name] = self.options.get(name,value)

        self.power_options = {'dk':0.01,'kmin':0.,'kmax':None,'los':'x','Nmu':3,'poles':[0,2,4]}
        for name,value in self.power_options.items():
            self.power_options[name] = self.options.get(name,value)
        if isinstance(self.power_options['los'],str):
            los = [0,0,0]
            los['xyz'.index(self.power_options['los'])] = 1
            self.power_options['los'] = los
        self.catalog_options = {'position':'Position'}
        for name,value in self.catalog_options.items():
            self.catalog_options[name] = self.options.get(name,value)

        self.data_load = self.options.get('data_load',None)

    def execute(self):
        data = utils.prepare_box_catalog(**self.catalog_options)
        data = data.to_nbodykit()

        mesh = data.to_mesh(position='position',**self.mesh_options)
        result = FFTPower(mesh,**self.power_options)
        attrs = result.attrs.copy()
        poles = result.poles
        ells = attrs['poles']
        proj = ['ell_{:d}'.format(ell) for ell in ells]
        x,y,proj = [],[],[]
        for ell in ells:
            x.append(poles['k'])
            if ell == 0:
                y.append(power['power_{:d}'.format(ell)].real - attrs['shotnoise'])
            else:
                y.append(power['power_{:d}'.format(ell)])
            proj.append('ell_{:d}'.format(ell))
        Nmu = attrs['Nmu']
        if Nmu is not None:
            for imu in range(Nmu):
                y.append(result.power['k'][:,imu])
                y.append(result.power['power'].real[:,imu] - attrs['shotnoise'])
                proj.append('mu_{:d}'.format(imu))
            attrs['muedges'] = muedges
        data_vector = DataVector(x=x,y=y,proj=proj,**attrs)
        if self.save: data_vector.save_auto(self.save)
        self.data_block[section_names.data,'data_vector'] = data_vector

    def cleanup(self):
        pass
