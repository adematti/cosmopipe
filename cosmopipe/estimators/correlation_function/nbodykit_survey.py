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

    def execute(self):
        data,randoms = prepare_survey_catalogs(data,randoms,cosmo=self.data_block.get(section_names.fiducial_cosmology,'cosmo',None),**self.catalog_options)
        data = data.to_nbodykit()
        randoms = randoms.to_nbodykit()

        class FakeCosmo(object):

            def comoving_distance(z):
                return np.ones_like(z)

        result = SurveyData2PCF(self.mode,data,randoms,self.edges,cosmo=FakeCosmo(),
                                data2=None,randoms2=None,R1R2=None,
                                ra='ra',dec='dec',redshift='distance',weight='weight',
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

    def cleanup(self):
        pass
