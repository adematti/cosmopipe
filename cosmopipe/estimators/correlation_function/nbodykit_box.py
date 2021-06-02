import logging

import numpy as np
from pypescript import BaseModule, ConfigError
from nbodykit.lab import SurveyData2PCF

from cosmopipe.lib.catalog import Catalog, utils
from .estimator import PairCount, NaturalEstimator


class BoxCorrelationFunction(BaseModule):

    logger = logging.getLogger('BoxCorrelationFunction')

    def setup(self):
        self.set_correlation_options()
        self.catalog_options = {'position':'Position'}
        for name,value in self.catalog_options.items():
            self.catalog_options[name] = self.options.get(name,value)
        self.data_load = self.options.get('data_load','data')

    def set_correlation_options(self):
        default_min = 1e-12 # non-zero to avoid cross-pairs
        self.correlation_options = {'mode':'2d','pimax':80,'edges':np.linspace(default_min,200,50),'muedges':100,'muwedges':3,'ells':(0,2,4),'show_progress':False,'nhtreads':1}
        for name,value in self.correlation_options.items():
            self.correlation_options[name] = self.options.get(name,value)
        self.mode = self.correlation_options.pop('mode')
        self.edges = self.correlation_options.pop('edges')
        self.ells = self.correlation_options.pop('ells')
        self.mode = self.correlation_options.pop('mode')
        if self.mode in ['rp','rppi']:
            self.correlation_options['mode'] = 'projected'
        else: # 1d, 2d, angular
            self.correlation_options['mode'] = self.mode
        self.muwedges = self.correlation_options.pop('muwedges')
        self.correlation_options['Nmu'] = self.correlation_options.pop('muedges')
        if isinstance(self.edges,dict):
            scale = self.edges.get('scale','lin')
            if scale == 'lin':
                self.edges = np.linspace(self.edges.get('min',default_min),self.edges['max'],self.edges['nbins']+1)
            elif scale == 'log':
                self.edges = np.logspace(np.log10(self.edges.get('min',default_min)),np.log10(self.edges['max']),self.edges['nbins']+1)
            else:
                raise ConfigError('Unknown scale {}'.format(scale))

    def build_data_vector(self, estimator):
        x,y,proj = [],[],[]
        if self.mode == '2d' and self.ells:
            s,poles = project_to_multipoles(estimator)
            x += [s]*len(self.ells)
            y += poles.T.tolist()
            proj += ['ell_{:d}'.format(ell) for ell in self.ells]
        if self.mode == '2d' and self.muwedges:
            estimator.rebin(edges=(estimator.edges[0],muedges))
            x += estimator.sep.T.tolist()
            y += estimator.corr.T.tolist()
            proj += ['mu_{:d}'.format(imu) for imu in range(len(self.muwedges)-1)]
        if self.mode == 'rp':
            dpi = np.diff(estimator.edges[1])
            wp = 2*(estimator.corr*dpi).sum(axis=-1)
            sep = estimator.sep[0].mean(axis=-1)
            x.append(sep)
            y.append(wp)
            proj.append('wp')
        if self.mode == 'rppi':
            x += estimator.sep.T.tolist()
            y += estimator.corr.T.tolist()
            proj += ['pi_{:d}'.format(imu) for imu in range(len(self.edges[-1])-1)]
        if self.mode == 'angular':
            x.append(estimator.sep)
            y.append(estimator.corr)
            proj.append('angular')
        if self.mode == '1d':
            x.append(estimator.sep)
            y.append(estimator.corr)
            proj.append('ell_0')
        return DataVector(x=x,y=y,proj=proj,**estimator.attrs)

    def execute(self):
        data = utils.prepare_box_catalog(**self.catalog_options)
        data = data.to_nbodykit()

        result = SimulationBox2PCF(self.mode,data,self.edges,randoms=None,
                                data2=None,randoms2=None,R1R2=None,
                                position='position',weight='weight',
                                **self.correlation_options)
        args = []
        for name in ['D1D2','R1R2']:
            pc = getattr(result,name)
            args.append(PairCount(wnpairs=pc['wnpairs'],total_wnpairs=pc['total_wnpairs']))
        edges = [result.edges[dim] for dim in result.dims]
        default_sep = np.meshgrid([(e[1:] + e[:-1])/2. for e in edges],indexing='ij')
        sep = []
        for idim,dim in enumerate(result.dims):
            s = result.R1R2['{}avg'.format(dim)]
            if np.isnan(s).all(): s = default_sep[idim]
            sep.append(s)
        estimator = NaturalEstimator(*args,edges=edges,sep=sep,**result.attrs)
        if self.save_estimator: estimator.save(self.save_estimator)
        data_vector = self.build_data_vector(estimator)
        if self.save: data_vector.save_auto(self.save)
        self.data_block[section_names.data,'data_vector'] = data_vector

    def cleanup(self):
        pass
