import logging

import numpy as np
from pypescript import BaseModule, ConfigError
from nbodykit.lab import SurveyData2PCF

from cosmopipe.lib.catalog import Catalog, utils
from .estimator import PairCount, LandySzalayEstimator


class SurveyCorrelationFunction(BoxCorrelationFunction):

    logger = logging.getLogger('SurveyCorrelationFunction')

    def setup(self):
        self.set_correlation_options()
        self.catalog_options = {'z':'Z','ra':'RA','dec':'DEC','position':None,'weight_comp':None,'nbar':{},'weight_fkp':None,'P0_fkp':0.}
        for name,value in self.catalog_options.items():
            self.catalog_options[name] = self.options.get(name,value)
        self.data_load = self.options.get('data_load','data')
        self.randoms_load = self.options.get('randoms_load','randoms')
        self.R1R2_load = self.options.get('R1R2_load',False)
        if isinstance(self.R1R2_load,bool) and self.R1R2_load:
            self.R1R2_load = 'correlation_estimator'

    def get_R1R2(self):
        R1R2 = None
        if self.R1R2_load:
            R1R2 = syntax.load_auto(self.R1R2_load,data_block=self.data_block,default_section=section_names.data,loader=NaturalEstimator.load)
        return R1R2

    def execute(self):
        input_data = syntax.load_auto(self.data_load,data_block=self.data_block,default_section=section_names.catalog,loader=Catalog.load_auto)
        input_randoms = syntax.load_auto(self.randoms_load,data_block=self.data_block,default_section=section_names.catalog,loader=Catalog.load_auto)
        if len(input_data) != len(input_randoms):
            raise ValueError('Number of input data and randoms catalogs is different ({:d} v.s. {:d})'.format(len(input_data),len(input_randoms)))
        list_data,list_randoms = [],[]
        for data,randoms in zip(input_data,input_randoms):
            if self.mode == 'angular':
                data,randoms = prepare_survey_angular_catalogs(data,randoms,**{name: self.catalog_options5[name] for name in ['ra','dec','weight_comp'])
            list_data.append(data.to_nbodykit())
            list_randoms.append(randoms.to_nbodykit())

        class FakeCosmo(object):

            def comoving_distance(z):
                return np.ones_like(z)

        cross = len(list_data) > 1
        result = SurveyData2PCF(self.mode,list_data[0],list_randoms[0],self.edges,cosmo=FakeCosmo(),
                                data2=list_data[1] if cross else None,randoms2=list_randoms[1] if cross else None,
                                R1R2=self.get_R1R2(),ra='ra',dec='dec',redshift='distance',weight='weight',
                                **self.correlation_options)
        args = []
        for name in ['D1D2','R1R2','D1R2','D2R1']:
            pc = getattr(result,name)
            args.append(PairCount(wnpairs=pc['wnpairs'],total_wnpairs=pc['total_wnpairs']))
        edges = [result.edges[dim] for dim in result.dims]
        default_sep = np.meshgrid([(e[1:] + e[:-1])/2. for e in edges],indexing='ij')
        sep = []
        for idim,dim in enumerate(result.dims):
            s = result.R1R2['{}avg'.format(dim)]
            if np.isnan(s).all(): s = default_sep[idim]
            sep.append(s)
        estimator = LandySzalayEstimator(*args,edges=edges,sep=sep,**result.attrs)
        if self.save_estimator: estimator.save(self.save_estimator)
        data_vector = self.build_data_vector(estimator)
        if self.save: data_vector.save_auto(self.save)
        self.data_block[section_names.data,'data_vector'] = data_vector
        self.data_block[section_names.data,'correlation_estimator'] = estimator

    def cleanup(self):
        pass
