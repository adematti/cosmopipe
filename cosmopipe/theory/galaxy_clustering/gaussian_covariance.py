import numpy as np
from scipy import special
from pypescript import SectionBlock

from cosmopipe import section_names
from cosmopipe.lib import syntax, utils
from cosmopipe.lib.data_vector import DataVector
from cosmopipe.lib.theory import gaussian_covariance
from cosmopipe.lib.theory.base import ProjectionBase
from cosmopipe.data_vector.data_vector import get_kwview


class GaussianCovariance(object):

    def setup(self):
        options = {}
        for name in ['volume','integration']:
            options[name] = self.options.get(name,None)
        x = self.options.get('x',None)
        edges = self.options.get('edges',None)
        projs = self.options.get('projs',None)
        if edges is not None and (isinstance(edges,dict) or np.ndim(edges[0]) == 0):
            edges = [edges]*len(projs)
        if isinstance(edges,list):
            if isinstance(edges[0],dict):
                edges = [utils.customspace(**kw) for kw in edges]
        self.kwview = {}
        data_load = self.options.get('data_load',edges is None)
        if data_load:
            if isinstance(data_load,bool) and data_load:
                data_load = 'data_vector'
            data_vector = self.data_block[syntax.split_sections(data_load,default_section=section_names.data)]
            data = data_vector.copy(copy_proj=True)
            data.data = [data.get(proj) for proj in data_vector.get_projs()]
            if projs is not None:
                data.view(proj=projs)
            if edges is not None:
                for proj,edges in zip(projs,edges):
                    dataproj = data.get(proj)
                    dataproj.edges[dataproj.attrs['x']] = edges
            self.kwview = data.kwview
            data = data.noview() # required for now, because covariance matrix without view should have data vectors without view
        else:
            if x is None:
                x = [(edge[:-1] + edge[1:])/2. for edge in edges]
            data = DataVector(x=x,proj=projs,edges=[{'x':edge} for edge in edges])
        model_bases = self.data_block[section_names.model,'collection'].bases
        self.covariance = gaussian_covariance.GaussianCovarianceMatrix(data=data,model_bases=model_bases,**options)

    def execute(self):
        collection = self.data_block[section_names.model,'collection']
        self.covariance.compute(collection)
        xlim = self.options.get('xlim',None)
        if xlim is not None:
            kwview = get_kwview(self.covariance.x[0],xlim=xlim)
        else:
            kwview = self.kwview
        cov = self.covariance.view(**kwview)
        self.data_block[section_names.covariance,'covariance_matrix'] = cov
        self.data_block[section_names.covariance,'cov'] = cov.get_cov()
        self.data_block[section_names.covariance,'invcov'] = cov.get_invcov()
        self.data_block[section_names.covariance,'nobs'] = None

    def cleanup(self):
        pass
