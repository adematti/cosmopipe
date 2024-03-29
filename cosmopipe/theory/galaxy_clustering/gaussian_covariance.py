import numpy as np
from scipy import special
from pypescript import SectionBlock

from cosmopipe import section_names
from cosmopipe.lib import syntax, utils
from cosmopipe.lib.data_vector import DataVector
from cosmopipe.lib.theory import gaussian_covariance
from cosmopipe.lib.theory.base import ProjectionBasis
from cosmopipe.data_vector.data_vector import get_kwview


class GaussianCovariance(object):

    def setup(self):
        options = {}
        for name in ['volume','integration']:
            options[name] = self.options.get(name,None)
        self.kwview = {}
        data_vector = self.data_block[section_names.data,'data_vector']
        model_bases = self.data_block[section_names.model,'collection'].bases()

        if not options['volume']:
            fw=self.data_block['data']['data_vector']
            options['volume']=fw.get(fw.projs[0]).attrs['Veff']
        self.covariance = gaussian_covariance.GaussianCovarianceMatrix(data_vector=data_vector,model_bases=model_bases,**options)

    def execute(self):
        collection = self.data_block[section_names.model,'collection']
        self.covariance.compute(collection)
        xlim = self.options.get('xlim',None)
        if xlim is not None:
            kwview = get_kwview(self.covariance.x[0],xlim=xlim)
        else:
            kwview = self.kwview
        cov = self.covariance.view(**kwview)
        save = self.options.get('save',None)
        if save: cov.save_auto(save)
        #print('COV',cov.get_x()[0],'y',cov.get_y()[0])
        #exit()
        self.data_block[section_names.covariance,'covariance_matrix'] = cov
        self.data_block[section_names.covariance,'cov'] = cov.get_cov()
        self.data_block[section_names.covariance,'invcov'] = cov.get_invcov()
        self.data_block[section_names.covariance,'nobs'] = None

    def cleanup(self):
        pass
