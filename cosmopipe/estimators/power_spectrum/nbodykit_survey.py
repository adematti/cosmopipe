import logging

from pypescript import BaseModule
from nbodykit.lab import FKPCatalog, ConvolvedFFTPower

from cosmopipe.lib.catalog import Catalog, utils


class SurveyPowerSpectrum(BaseModule):

    logger = logging.getLogger('SurveyPowerSpectrum')

    def setup(self):
        self.mesh_options = {'Nmesh':512,'BoxSize':None,'BoxPad':0.02,'resampler':'tsc','interlaced':True}
        for name,value in self.mesh_options.items():
            self.mesh_options[name] = self.options.get(name,value)

        self.power_options = {'dk':0.01,'kmin':0.,'kmax':None,'muwedges':None,'ells':(0,2,4)}
        for name,value in self.power_options.items():
            self.power_options[name] = self.options.get(name,value)
        self.ells = self.power_options.pop('ells')
        self.muwedges = self.power_options.pop('muwedges',None)

        self.catalog_options = {'z':'Z','ra':'RA','dec':'DEC','position':None,'weight_comp':None,'nbar':{},'weight_fkp':None,'P0_fkp':0.}
        for name,value in self.catalog_options.items():
            self.catalog_options[name] = self.options.get(name,value)
        self.data_load = self.options.get('data_load','data')
        self.randoms_load = self.options.get('randoms_load','randoms')

    def execute(self):
        input_data = syntax.load_auto(self.data_load,data_block=self.data_block,default_section=section_names.catalog,loader=Catalog.load_auto)
        input_randoms = syntax.load_auto(self.randoms_load,data_block=self.data_block,default_section=section_names.catalog,loader=Catalog.load_auto)
        if len(input_data) != len(input_randoms):
            raise ValueError('Number of input data and randoms catalogs is different ({:d} v.s. {:d})'.format(len(input_data),len(input_randoms)))
        list_mesh = []
        for data,randoms in zip(input_data,input_randoms):
            data,randoms = prepare_survey_catalogs(data,randoms,cosmo=self.data_block.get(section_names.fiducial_cosmology,'cosmo',None),**self.catalog_options)
            fkp = FPKCatalog(data.to_nbodykit(),randoms.to_nbodykit())
            list_mesh.append(fpk.to_mesh(position='position',fkp_weight='weight_fkp',comp_weight='weight_comp',nbar='nbar',**self.mesh_options))

        result = ConvolvedFFTPower(list_mesh[0],poles=self.ells,second=list_mesh[1] if len(list_meshs) > 1 else None,**self.power_options)
        attrs = result.attrs.copy()
        poles = result.poles
        y,mapping_proj = [],[]
        for ell in self.ells:
            if ell == 0:
                y.append(power['power_{:d}'.format(ell)].real - attrs['shotnoise'])
            else:
                y.append(power['power_{:d}'.format(ell)])
            mapping_proj.append('ell_{:d}'.format(ell))
        if self.muwedges is not None:
            pkmu = result.to_pkmu(self.muedges,ells[-1])
            for imu,(low,up) in enumerate(zip(self.muwedges[:-1],self.muwedges[1:])):
                y.append(pkmu['power'].real[:,imu] - attrs['shotnoise'])
                mapping_proj.append(('muwedge',(low,up)))
        data_vector = DataVector(x=poles['k'],y=y,mapping_proj=mapping_proj,**attrs)
        if self.save: data_vector.save_auto(self.save)
        self.data_block[section_names.data,'data_vector'] = data_vector

    def cleanup(self):
        pass
