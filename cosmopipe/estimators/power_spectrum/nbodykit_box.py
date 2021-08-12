import logging

import numpy as np
from nbodykit.lab import FFTPower
from pypescript import BaseModule

from cosmopipe import section_names
from cosmopipe.lib import syntax
from cosmopipe.lib.catalog import Catalog
from cosmopipe.lib.data_vector import DataVector, ProjectionName, BinnedProjection
from cosmopipe.estimators import utils


class BoxPowerSpectrum(BaseModule):

    logger = logging.getLogger('BoxPowerSpectrum')

    def setup(self):
        self.mesh_options = {'Nmesh':512,'BoxSize':None,'resampler':'tsc','interlaced':True,'compensated':True}
        for name,value in self.mesh_options.items():
            self.mesh_options[name] = self.options.get(name,value)
        self.BoxSize = self.mesh_options.pop('BoxSize',None)
        self.power_options = {'los':'x','muwedges':3,'ells':(0,2,4)}
        for name,value in self.power_options.items():
            self.power_options[name] = self.options.get(name,value)
        kedges = self.options.get('edges',{})
        self.power_options['kmin'] = kedges.get('min',0.)
        self.power_options['kmax'] = kedges.get('max',None)
        self.power_options['dk'] = kedges.get('step',None)
        if isinstance(self.power_options['los'],str):
            los = [0,0,0]
            los['xyz'.index(self.power_options['los'])] = 1
            self.power_options['los'] = los
        self.ells = self.power_options.pop('ells')
        self.muwedges = self.power_options['Nmu'] = self.power_options.pop('muwedges')
        self.muwedges = np.linspace(0.,1.,self.muwedges+1)
        self.catalog_options = {'position':'Position'}
        for name,value in self.catalog_options.items():
            self.catalog_options[name] = self.options.get(name,value)

    def execute(self):
        self.data_load = self.options.get('data_load','data')
        self.projattrs = self.options.get('projattrs',{})
        if isinstance(self.projattrs,str):
            self.projattrs = {'name':self.projattrs}
        self.save = self.options.get('save',None)
        input_data = syntax.load_auto(self.data_load,data_block=self.data_block,default_section=section_names.catalog,loader=Catalog.load_auto)
        list_mesh = []
        BoxSize = self.BoxSize
        for data in input_data:
            if BoxSize is None: BoxSize = data.attrs.get('BoxSize',None)
        for data in input_data:
            mesh = utils.prepare_box_catalog(data,**self.catalog_options).to_nbodykit().to_mesh(position='position',BoxSize=self.BoxSize,**self.mesh_options)
            list_mesh.append(mesh)

        result = FFTPower(list_mesh[0],mode='2d',second=list_mesh[1] if len(list_mesh) > 1 else None,poles=self.ells,**self.power_options)
        attrs = result.attrs.copy()
        poles = result.poles
        ells = attrs['poles']
        data_vector = DataVector()
        for ell in ells:
            x = poles['k']
            if ell == 0:
                y = poles['power_{:d}'.format(ell)] - attrs['shotnoise']
            else:
                y = poles['power_{:d}'.format(ell)]
            y = y.real if ell % 2 == 0 else y.imag
            proj = ProjectionName(space=ProjectionName.POWER,mode=ProjectionName.MULTIPOLE,proj=ell,**self.projattrs)
            dataproj = BinnedProjection(data={'k':x,'power':y,'nmodes':poles['modes']},x='k',y='power',weights='nmodes',edges={'k':result.poles.edges['k']},proj=proj,attrs=attrs)
            data_vector.set(dataproj)
        pkmu = result.power
        for imu,(low,up) in enumerate(zip(self.muwedges[:-1],self.muwedges[1:])):
            x = pkmu['k'][:,imu]
            y = pkmu['power'].real[:,imu] - attrs['shotnoise']
            proj = ProjectionName(space=ProjectionName.POWER,mode=ProjectionName.MUWEDGE,proj=(low,up),**self.projattrs)
            dataproj = BinnedProjection(data={'k':x,'power':y,'nmodes':poles['modes']},x='k',y='power',weights='nmodes',edges={'k':result.poles.edges['k']},proj=proj,attrs=attrs)
            data_vector.set(dataproj)

        if self.save: data_vector.save_auto(self.save)
        self.data_block[section_names.data,'data_vector'] = self.data_block.get(section_names.data,'data_vector',[]) + data_vector

    def cleanup(self):
        pass
