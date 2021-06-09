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

        self.power_options = {'dk':0.01,'kmin':0.,'kmax':None,'los':'x','muwedges':3,'ells':(0,2,4)}
        for name,value in self.power_options.items():
            self.power_options[name] = self.options.get(name,value)
        if isinstance(self.power_options['los'],str):
            los = [0,0,0]
            los['xyz'.index(self.power_options['los'])] = 1
            self.power_options['los'] = los
        self.ells = self.power_options.pop('ells')
        self.muwedges = self.power_options['Nmu'] = self.power_options.pop('muwedges')
        self.catalog_options = {'position':'Position'}
        for name,value in self.catalog_options.items():
            self.catalog_options[name] = self.options.get(name,value)
        self.data_load = self.options.get('data_load','data')

    def execute(self):
        input_data = syntax.load_auto(self.data_load,data_block=self.data_block,default_section=section_names.catalog,loader=Catalog.load_auto)
        list_mesh = []
        for data in input_data:
            mesh = utils.prepare_box_catalog(data,**self.catalog_options).to_nbodykit().to_mesh(position='position',**self.mesh_options)
            list_mesh.append(mesh)

        result = FFTPower(list_mesh[0],mode='2d',second=list_mesh[1] if len(list_meshs) > 1 else None,poles=self.ells,**self.power_options)
        attrs = result.attrs.copy()
        poles = result.poles
        ells = attrs['poles']
        x,y,mapping_proj = [],[],[]
        for ell in ells:
            x.append(poles['k'])
            if ell == 0:
                y.append(power['power_{:d}'.format(ell)].real - attrs['shotnoise'])
            else:
                y.append(power['power_{:d}'.format(ell)])
            mapping_proj.append('ell_{:d}'.format(ell))

        pkmu = result.power
        for imu,(low,up) in enumerate(zip(self.muwedges[:-1],self.muwedges[1:])):
            y.append(pkmu['power'].real[:,imu] - attrs['shotnoise'])
            mapping_proj.append(('muwedge',(low,up)))

        data_vector = DataVector(x=x,y=y,mapping_proj=mapping_proj,**attrs)
        if self.save: data_vector.save_auto(self.save)
        self.data_block[section_names.data,'data_vector'] = data_vector

    def cleanup(self):
        pass
