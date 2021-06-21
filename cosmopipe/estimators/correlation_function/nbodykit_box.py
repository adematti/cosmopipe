import logging

import numpy as np
from pypescript import BaseModule, ConfigError
from nbodykit.lab import SimulationBox2PCF

from cosmopipe import section_names
from cosmopipe.lib import syntax
from cosmopipe.lib.catalog import Catalog
from cosmopipe.lib.data import DataVector
from cosmopipe.lib.utils import customspace, dict_nonedefault
from cosmopipe.estimators import utils
from cosmopipe.lib.estimators.correlation_function import PairCount, NaturalEstimator, project_to_multipoles


class BoxCorrelationFunction(BaseModule):

    logger = logging.getLogger('BoxCorrelationFunction')

    def setup(self):
        self.set_correlation_options()
        self.catalog_options = {'position':'Position'}
        for name,value in self.catalog_options.items():
            self.catalog_options[name] = self.options.get(name,value)
        self.BoxSize = self.options.get('BoxSize')
        self.data_load = self.options.get('data_load','data')
        self.save = self.options.get('save',None)
        self.save_estimator = self.options.get('save_estimator',None)

    def set_correlation_options(self):
        default_edges = {'min':1e-12,'max':200,'nbins':50} # non-zero to avoid cross-pairs
        self.correlation_options = {'mode':'2d','pimax':80,'muedges':100,'muwedges':3,'ells':(0,2,4),'show_progress':False,'nthreads':1}
        for name,value in self.correlation_options.items():
            self.correlation_options[name] = self.options.get(name,value)
        self.mode = self.correlation_options.pop('mode')
        edges = self.options.get('edges',default_edges)
        self.edges = customspace(**dict_nonedefault(edges,**default_edges))
        self.ells = self.correlation_options.pop('ells')
        if self.mode in ['rp','rppi']:
            self.nbodykit_mode = 'projected'
        else: # 1d, 2d, angular
            self.nbodykit_mode = self.mode
            self.correlation_options.pop('pimax')
        self.muwedges = self.correlation_options.pop('muwedges')
        if np.ndim(self.muwedges) == 0:
            self.muwedges = np.linspace(0.,1.,self.muwedges+1)
        self.correlation_options['Nmu'] = self.correlation_options.pop('muedges')

    def build_data_vector(self, estimator):
        x,y,mapping_proj = [],[],[]
        if self.mode == '2d' and self.ells:
            s,poles = project_to_multipoles(estimator)
            x += [s]*len(self.ells)
            y += poles.T.tolist()
            mapping_proj += ['ell_{:d}'.format(ell) for ell in self.ells]
        if self.mode == '2d' and self.muwedges.size:
            estimator.rebin((estimator.edges[0],self.muwedges))
            x += estimator.sep[0].T.tolist()
            y += estimator.corr.T.tolist()
            mapping_proj += [('muwedge',(low,up)) for low,up in zip(self.muwedges[:-1],self.muwedges[1:])]
        if self.mode == 'rp':
            dpi = np.diff(estimator.edges[1])
            wp = 2*(estimator.corr*dpi).sum(axis=-1)
            sep = estimator.sep[0].mean(axis=-1)
            x.append(sep)
            y.append(wp)
            mapping_proj = None
        if self.mode == 'rppi':
            x += estimator.sep.T.tolist()
            y += estimator.corr.T.tolist()
            mapping_proj += [('piwedge',(low,up)) for low,up in zip(estimator.edges[1][:-1],estimator.edges[1][1:])]
        if self.mode == 'angular':
            x.append(estimator.sep)
            y.append(estimator.corr)
            mapping_proj = None
        if self.mode == '1d':
            x.append(estimator.sep)
            y.append(estimator.corr)
            mapping_proj = None
        return DataVector(x=x,y=y,mapping_proj=mapping_proj,**estimator.attrs)

    def execute(self):
        input_data = syntax.load_auto(self.data_load,data_block=self.data_block,default_section=section_names.catalog,loader=Catalog.load_auto)
        list_data = []
        for data in input_data:
            data = utils.prepare_box_catalog(data,**self.catalog_options).to_nbodykit()
            list_data.append(data)
        BoxSize = self.BoxSize
        for data in input_data:
            if BoxSize is None: BoxSize = data.attrs.get('BoxSize',None)
        result = SimulationBox2PCF(self.nbodykit_mode,list_data[0],self.edges,
                                data2=list_data[1] if len(list_data) > 1 else None,
                                R1R2=None,
                                position='position',weight='weight',BoxSize=BoxSize,
                                **self.correlation_options)
        args = []
        for name in ['D1D2','R1R2']:
            pc = getattr(result,name)
            args.append(PairCount(wnpairs=pc['wnpairs'],total_wnpairs=pc.attrs['total_wnpairs']))
        edges = [result.corr.edges[dim] for dim in result.corr.dims]
        default_sep = np.meshgrid(*[(e[1:] + e[:-1])/2. for e in edges],indexing='ij')
        sep = []
        for idim,dim in enumerate(result.corr.dims):
            if '{}avg'.format(dim) in result.R1R2 and not np.isnan(result.R1R2['{}avg'.format(dim)]).all():
                s = result.R1R2['{}avg'.format(dim)]
            else: s = default_sep[idim]
            sep.append(s)
        result.attrs.pop('edges')
        estimator = NaturalEstimator(*args,edges=edges,sep=sep,**result.attrs)
        if self.save_estimator: estimator.save(self.save_estimator)
        data_vector = self.build_data_vector(estimator)
        if self.save: data_vector.save_auto(self.save)
        self.data_block[section_names.data,'data_vector'] = data_vector
        self.data_block[section_names.data,'correlation_estimator'] = estimator

    def cleanup(self):
        pass
