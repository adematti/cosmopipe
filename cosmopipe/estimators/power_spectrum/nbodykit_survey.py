import logging

import os
import numpy as np
from pypescript import BaseModule
from nbodykit.lab import FKPCatalog, ConvolvedFFTPower

from cosmopipe import section_names
from cosmopipe.lib import syntax, mpi
from cosmopipe.lib.catalog import Catalog
from cosmopipe.lib.data_vector import DataVector, ProjectionName, BinnedProjection
from cosmopipe.estimators import utils


def is_valid_crosscorr(*args, **kwargs):
    return True


# temporary patch
from nbodykit.algorithms.convpower import fkp
fkp.is_valid_crosscorr = is_valid_crosscorr


class SurveyPowerSpectrum(BaseModule):

    logger = logging.getLogger('SurveyPowerSpectrum')

    def setup(self):
        self.mesh_options = {'Nmesh':512,'BoxSize':None,'BoxPad':0.02,'resampler':'tsc','interlaced':True}
        for name,value in self.mesh_options.items():
            self.mesh_options[name] = self.options.get(name,value)
        self.BoxPad = self.mesh_options.pop('BoxPad')

        self.power_options = {'muwedges':None,'ells':(0,2,4)}
        for name,value in self.power_options.items():
            self.power_options[name] = self.options.get(name,value)
        kedges = self.options.get('edges',{})
        self.power_options['kmin'] = kedges.get('min',0.)
        self.power_options['kmax'] = kedges.get('max',None)
        self.power_options['dk'] = kedges.get('step',None)
        self.ells = self.power_options.pop('ells')
        self.muwedges = self.power_options.pop('muwedges',None)
        if self.muwedges is not None and np.ndim(self.muwedges) == 0:
            self.muwedges = np.linspace(0.,1.,self.muwedges+1)

        self.catalog_options = {'z':'Z','ra':'RA','dec':'DEC','position':None,'weight_comp':None,'nbar':{},'weight_fkp':None,'P0_fkp':0.,'zmin':0,'zmax':10.,'ramax':400.}
        for name,value in self.catalog_options.items():
            self.catalog_options[name] = self.options.get(name,value)
        self.projattrs = self.options.get('projattrs',{})
        if isinstance(self.projattrs,str):
            self.projattrs = {'name':self.projattrs}

    def execute(self):
        self.save = self.options.get('save',None)
        if not self.save:
            #no default so it is possible to not print, as window tries to do by default
            self.save = self.options.get('saveroot',None)
            if self.save: 
                self.save+=str(self.catalog_options['zmin'])+"_"+str(self.catalog_options['zmax'])+"_"+\
                       str(self.mesh_options['BoxSize'])+"_"+str(self.mesh_options['Nmesh'])
                if self.power_options['dk']: 
                    self.save+="_"+str(self.power_options['dk'])
                self.save+=".txt"
        self.use_existing = self.options.get('use_existing',None)
        if self.use_existing and os.path.isfile(self.save) :
            loaddv =  DataVector.load_auto(self.save)
            self.data_block[section_names.data,'data_vector'] = self.data_block.get(section_names.data,'data_vector',[]) + loaddv
            return
        self.data_load = self.options.get('data_load','data')
        self.randoms_load = self.options.get('randoms_load','randoms')
        input_data = syntax.load_auto(self.data_load,data_block=self.data_block,default_section=section_names.catalog,loader=Catalog.load_auto)
        has_randoms = self.randoms_load != ''
        if has_randoms:
            input_randoms = syntax.load_auto(self.randoms_load,data_block=self.data_block,default_section=section_names.catalog,loader=Catalog.load_auto)
            if len(input_data) != len(input_randoms):
                raise ValueError('Number of input data and randoms catalogs is different ({:d} v.s. {:d})'.format(len(input_data),len(input_randoms)))
        else:
            self.log_info('Using no randoms (e.g., for window).',rank=0)
            input_randoms = [None]*len(input_data)
        cosmo = self.data_block.get(section_names.fiducial_cosmology,'cosmo',None)
        list_mesh = []
        wdata2 = 1.
        zeffdata = 0.
        zeffran = 0.
        wdata = 0.
        wran = 0.
        Gausstestfile=self.options.get('Gausstestfile',None)
        if Gausstestfile and has_randoms :
          ranfield=np.fromfile(Gausstestfile,dtype='f4')    
        for data,randoms in zip(input_data,input_randoms):
            data = data.mpi_to_state('scattered')
            if randoms is not None: randoms = randoms.mpi_to_state('scattered')
            data,randoms = utils.prepare_survey_catalogs(data,randoms,cosmo=cosmo,**self.catalog_options)
            #clear stuff from data fits files added here (Arnaud suggested placement) 
            data.attrs = {}
            if randoms : randoms.attrs = {}
            fkp = FKPCatalog(data.to_nbodykit(),randoms.to_nbodykit() if randoms is not None else None,BoxPad=self.BoxPad,nbar='nbar')
            if Gausstestfile and randoms is not None :
              L=self.mesh_options['BoxSize']
              N=self.mesh_options['Nmesh']
              BoxCenter=fkp._define_bbox('position','Selection', "randoms")[1]
              xind=np.floor((randoms['position'][:,0]-BoxCenter[0]+L/2.0)/(L/N)).astype(int)
              yind=np.floor((randoms['position'][:,1]-BoxCenter[1]+L/2.0)/(L/N)).astype(int)
              zind=np.floor((randoms['position'][:,2]-BoxCenter[2]+L/2.0)/(L/N)).astype(int)
              alpha= mpi.sum_array(data['weight_comp'],mpicomm=data.mpicomm)/mpi.sum_array(randoms['weight_comp'],mpicomm=data.mpicomm)
              #print('alpha=',alpha)
              ind=zind+yind*N+xind*N*N
              data=randoms.deepcopy()
              data['weight_comp']*=alpha*(1+ranfield[ind])
              fkp = FKPCatalog(data.to_nbodykit(),randoms.to_nbodykit(),BoxPad=self.BoxPad,nbar='nbar')
            list_mesh.append(fkp.to_mesh(position='position',fkp_weight='weight_fkp',comp_weight='weight_comp',nbar='nbar',**self.mesh_options))
            #what is going on here with the multiplication when there is more than one catalog?!?
            wdata2 *= mpi.sum_array(data['weight_comp']*data['weight_fkp'],mpicomm=data.mpicomm)
            wdata = mpi.sum_array(data['weight_comp']*data['weight_fkp'],mpicomm=data.mpicomm)
            zeffdata = mpi.sum_array(data['z']*data['weight_comp']*data['weight_fkp'],mpicomm=data.mpicomm)
            zeffdata /= wdata
            if randoms is not None:
                wran*= mpi.sum_array(randoms['weight_comp']*randoms['weight_fkp'],mpicomm=data.mpicomm)
                zeffran = mpi.sum_array(randoms['z']*randoms['weight_comp']*randoms['weight_fkp'],mpicomm=data.mpicomm)
                zeffran /= wran
                    
        if len(list_mesh) == 1:
            wdata2 **= 2
        result = ConvolvedFFTPower(list_mesh[0],poles=self.ells,second=list_mesh[1] if len(list_mesh) > 1 else None,**self.power_options)
        attrs = result.attrs.copy()
        attrs['norm/wdata2'] = attrs['randoms.norm'] / wdata2
        attrs['zeffdata'] = zeffdata
        attrs['zeffran'] = zeffran
        if len(list_mesh) == 1:
            attrs['Veff'] = attrs['data.norm']*self.catalog_options['P0_fkp']**2 
       
        poles = result.poles
        ells = attrs['poles']
        shotnoise = attrs['shotnoise']
        shotnoise = shotnoise if np.isfinite(shotnoise) else 0.
        data_vector = DataVector()
        for ell in ells:
            x = poles['k']
            if ell == 0:
                y = poles['power_{:d}'.format(ell)] - shotnoise
            else:
                y = poles['power_{:d}'.format(ell)]
            y = y.real if ell % 2 == 0 else y.imag
            proj = ProjectionName(space=ProjectionName.POWER,mode=ProjectionName.MULTIPOLE,proj=ell,**self.projattrs)
            dataproj = BinnedProjection(data={'k':x,'power':y,'nmodes':poles['modes']},x='k',y='power',weights='nmodes',edges={'k':result.poles.edges['k']},proj=proj,attrs=attrs)
            data_vector.set(dataproj)
        if self.muwedges is not None:
            pkmu = result.to_pkmu(self.muwedges,self.ells[-1])
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
