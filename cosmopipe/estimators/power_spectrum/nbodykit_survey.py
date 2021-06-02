import logging

from pypescript import BaseModule
from nbodykit.lab import FKPCatalog, ConvolvedFFTPower

from cosmopipe.lib.catalog import Catalog, utils


class SurveyPowerSpectrum(BaseModule):

    logger = logging.getLogger('SurveyPowerSpectrum')

    def setup(self):
        self.mesh_options = {'Nmesh':512,'BoxSize':None,'resampler':'tsc','interlaced':True}
        for name,value in self.mesh_options.items():
            self.mesh_options[name] = self.options.get(name,value)

        self.power_options = {'dk':0.01,'kmin':0.,'kmax':None,'Nmu':None,'ells':(0,2,4)}
        for name,value in self.power_options.items():
            self.power_options[name] = self.options.get(name,value)
        self.power_options['poles'] = self.power_options.pop('ells') 

        self.catalog_options = {'z':'Z','ra':'RA','dec':'DEC','position':None,'weight_comp':None,'nbar':{},'weight_fkp':None,'P0_fkp':0.}
        for name,value in self.catalog_options.items():
            self.catalog_options[name] = self.options.get(name,value)
        self.Nmu = self.power_options.pop('Nmu',None)
        self.data_load = self.options.get('data_load',None)
        self.randoms_load = self.options.get('randoms_load',None)

    def compute_catalogs(self):

        origin_catalogs = {'data':data,'randoms':randoms}
        catalogs = {name:catalog.copy(columns=[]) for name,catalog in origin_catalogs.items()}

        def from_origin(column):
            for name,catalog in catalogs.items():
                catalog[name] = origin_catalogs.eval(self.catalog_options[name])

        if self.catalog_options['weight_comp'] is None:
            for name,catalog in catalogs.items():
                catalog['weight_comp'] = origin_catalogs[name].ones()
        else:
            from_origin('weight_comp')

        if self.catalog_options['z'] is not None:
            from_origin('z')

        if self.catalog_options['position'] is None:
            cosmo = self.data_block[section_names.fiducial_cosmology,'cosmo']
            for name,catalog in catalogs.items():
                distance = cosmo.comoving_radial_distance(catalog['z'])
                catalog['position'] = utils.sky_to_cartesian(distance,origin_catalogs[self.catalog_options['ra']],origin_catalogs[self.catalog_options['dec']],degree=True)
        else:
            from_origin('position')

        if isinstance(self.catalog_options['nbar'],dict):
            if 'z' in randoms:
                z = randoms['z']
                cosmo = self.data_block[section_names.fiducial_cosmology,'cosmo']
            else:
                z = utils.distance(randoms['position'])
                cosmo = None
            nbar = utils.RedshiftDensityInterpolator(redshifts,weights=randoms['weight_comp'],bins=self.catalog_options['nbar'],cosmo=cosmo,**randoms.mpi_attrs)
            for name,catalog in catalogs.items():
                if 'z' in randoms:
                    catalog['nbar'] = nbar(catalog['z'])
                else:
                    catalog['nbar'] = nbar(utils.distance(catalog['position']))
        else:
            from_origin('nbar')

        if self.catalog_options['weight_fkp'] is None:
            for name,catalog in catalogs.items():
                catalog['weight_fkp'] = 1./(1. + catalog['nbar']*self.catalog_options['P0_fkp'])
        else:
            from_origin('weight_fkp')
        return catalogs['data'],catalogs['randoms']

    def execute(self):
        data,randoms = self.compute_catalogs()
        if self.mesh_options.get('BoxSize',None) is None:
            BoxPad = np.array(self.mesh_options.get('BoxPad',0.02))
            BoxSize = (mpi.max_array(catalog['position'],axis=0) - mpi.min_array(catalog['position'],axis=0))*(1. + BoxPad)
            self.log_info('No BoxSize provided, found {} from randoms catalog'.format(BoxSize),rank=0)

        for name,catalog in catalogs.items():
            catalogs[name] = catalog.to_nbodykit()

        fkp = FPKCatalog(data,randoms)
        mesh = fkp.to_mesh(position='position',fkp_weight='weight_fkp',comp_weight='weight_comp',nbar='nbar',**self.mesh_options)
        result = ConvolvedFFTPower(mesh,**self.power_options)
        attrs = result.attrs.copy()
        poles = result.poles
        ells = attrs['poles']
        proj = ['ell_{:d}'.format(ell) for ell in ells]
        y,proj = [],[]
        for ell in ells:
            if ell == 0:
                y.append(power['power_{:d}'.format(ell)].real - attrs['shotnoise'])
            else:
                y.append(power['power_{:d}'.format(ell)])
            proj.append('ell_{:d}'.format(ell))
        if self.Nmu is not None:
            muedges = np.linspace(0,1,self.Nmu+1)
            pkmu = result.to_pkmu(muedges,ells[-1])
            for imu in range(self.Nmu):
                y.append(pkmu['power'].real[:,imu] - attrs['shotnoise'])
                proj.append('mu_{:d}'.format(imu))
            attrs['muedges'] = muedges
        data_vector = DataVector(x=poles['k'],y=y,proj=proj,**attrs)
        if self.save: data_vector.save_auto(self.save)
        self.data_block[section_names.data,'data_vector'] = data_vector

    def cleanup(self):
        pass
