import logging

import numpy as np
from pypescript import BasePipeline
from nbodykit.lab import FKPCatalog, ConvolvedFFTPower

from cosmopipe import section_names
from cosmopipe.lib import syntax, mpi, utils
from cosmopipe.lib.catalog import Catalog
from cosmopipe.lib.data_vector import DataVector, ProjectionName, BinnedProjection
from cosmopipe.lib.survey_selection import WindowFunction
from cosmopipe.estimators.utils import prepare_survey_catalogs


class FFTWindowFunction(BasePipeline):

    logger = logging.getLogger('FFTWindowFunction')

    def setup(self):
        self.swin = self.options.get('swin',None)
        if isinstance(self.swin,dict):
            self.swin = utils.customspace(**self.swin)
        if self.swin is not None: self.swin = np.asarray(self.swin)
        self.wa_orders = self.options.get('wa_orders',0)
        if np.ndim(self.wa_orders) == 0: self.wa_orders = [self.wa_orders]
        self.ells = self.options.get('ells',[0,2,4])
        if np.ndim(self.ells[0]) == 0: self.ells = [self.ells]*len(self.wa_orders)
        self.catalog_options = {'z':'Z','ra':'RA','dec':'DEC','position':None,'weight_comp':None,'nbar':{},'weight_fkp':None,'P0_fkp':0.}
        for name,value in self.catalog_options.items():
            self.catalog_options[name] = self.options.get(name,value)

    def execute(self):
        self.randoms_load = self.options.get('randoms_load','randoms')
        self.save = self.options.get('save',None)
        input_randoms = syntax.load_auto(self.randoms_load,data_block=self.data_block,default_section=section_names.catalog,loader=Catalog.load_auto,mpistate='scattered')

        cosmo = self.data_block.get(section_names.fiducial_cosmology,'cosmo',None)
        norm = self.options.get('norm',None)
        list_randoms = []
        if norm is None:
            self.data_load = self.options.get('data_load','data')
            input_data = syntax.load_auto(self.data_load,data_block=self.data_block,default_section=section_names.catalog,loader=Catalog.load_auto,mpistate='scattered')
            if len(input_data) != len(input_randoms):
                raise ValueError('Number of input data and randoms catalogs is different ({:d} v.s. {:d})'.format(len(input_data),len(input_randoms)))
            weight_fkp = []
            wdata2 = 1.
            for data,randoms in zip(input_data,input_randoms):
                data,randoms = prepare_survey_catalogs(data,randoms,cosmo=cosmo,**self.catalog_options)
                alpha = data.sum('weight_comp')/randoms.sum('weight_comp') # not total weight to mimic what is in nbodykit
                weight_fkp.append(randoms['weight_fkp'])
                list_randoms.append(randoms)
                wdata2 *= mpi.sum_array(data['weight_comp']*data['weight_fkp'],mpicomm=data.mpicomm)
            if len(weight_fkp) == 1:
                weight_fkp.append(weight_fkp[0])
                wdata2 **= 2
            norm = alpha * mpi.sum_array(randoms['nbar']*randoms['weight_comp']*weight_fkp[0]*weight_fkp[1],mpicomm=randoms.mpicomm)
            norm /= wdata2
        elif isinstance(norm,str):
            norm = self.data_block[section_names.data,'data_vector'].attrs[norm]
            list_randoms = [prepare_survey_catalogs(randoms,cosmo=cosmo,**self.catalog_options) for randoms in list_randoms]
            randoms = list_randoms[-1]

        is_auto = len(list_randoms) == 1 or list_randoms[1] is list_randoms[0]
        if len(list_randoms) == 1: list_randoms.append(list_randoms[0].copy())
        norm *= np.prod([mpi.sum_array(randoms['weight_comp']*randoms['weight_fkp'],mpicomm=randoms.mpicomm) for randoms in list_randoms])
        list_name_randoms = []
        self.pipe_block = self.data_block.copy()

        for irandoms,randoms in enumerate(list_randoms):
            list_name_randoms.append('randoms_{:d}'.format(irandoms))
            self.pipe_block[section_names.catalog,list_name_randoms[-1]] = randoms

        randoms = list_randoms[-1]
        origin_weights = randoms['weight_fkp'].copy()
        fourier_window = WindowFunction()

        def distance(position):
            return np.sqrt(np.sum(position**2,axis=-1))

        for todo in self.execute_todos:
            options = dict(todo.module.options)
            todo.module.options['data_load'] = list_name_randoms
            todo.module.options['randoms_load'] = ''
            todo.module.options['save'] = None
            for name in ['position','weight_fkp','weight_comp','nbar']: # catalogs are already provided
                todo.module.options[name] = name
            for wa_order,ells in zip(self.wa_orders,self.ells):
                self.pipe_block[section_names.data,'data_vector'] = []
                todo.module.options['ells'] = ells
                todo.module.options['muwedges'] = None
                todo.set_data_block()
                todo.module.setup()
                randoms['weight_fkp'] = origin_weights/distance(randoms['position'])**wa_order
                if is_auto:
                    shotnoise = mpi.sum_array(list_randoms[0]['weight_comp']*list_randoms[0]['weight_fkp']*randoms['weight_comp']*randoms['weight_fkp'],mpicomm=randoms.mpicomm)
                else:
                    shotnoise = 0.
                todo()
                fw = self.pipe_block[section_names.data,'data_vector']
                fw.attrs['shotnoise'] = shotnoise
                for proj in fw.projs:
                    dataproj = fw.get(proj)
                    dataproj.attrs['shotnoise'] = shotnoise
                    if proj.mode == ProjectionName.MULTIPOLE and proj.proj == 0:
                        y = dataproj.get_y() - shotnoise
                    else:
                        y = dataproj.get_y()
                    dataproj.set_y(y/norm)
                for proj in fw.projs:
                    proj.wa_order = wa_order
                fourier_window.extend(fw)
            for name,value in options.items():
                todo.module.options[name] = value

        if self.swin is not None:
            from cosmopipe.lib.survey_selection.window_function import compute_real_window_1d
            fourier_window.extend(compute_real_window_1d(self.swin,fourier_window))

        if self.save: fourier_window.save_auto(self.save)

        self.data_block[section_names.survey_selection,'window'] = fourier_window
