import numpy as np
from pypescript import syntax

from cosmopipe import section_names
from cosmopipe.lib.data_vector import DataVector, DataPlotStyle
from cosmopipe.lib.theory import evaluation
from cosmopipe.lib import syntax, utils


class ModelPlotting(object):

    def setup(self):
        self.data_load = self.options.get('data_load',None)
        if isinstance(self.data_load,bool) and self.data_load:
            self.data_load = 'data_vector'
        self.xmodel = self.options.get('xmodel',None)
        self.covariance_load = self.options.get('covariance_load',None)
        if isinstance(self.covariance_load,bool) and self.covariance_load:
            self.covariance_load = 'covariance_matrix'
        if self.covariance_load:
            self.covariance_load = syntax.split_sections(self.covariance_load,default_section=section_names.covariance)
        linestyles = ['-']
        if self.data_load:
            linestyles.append('--')
        self.style = DataPlotStyle(**{'linestyles':linestyles,**syntax.remove_keywords(self.options)})
        self.save_model = self.options.get('save_model',None)
        self.projection_collection = self.data_block[section_names.survey_selection,'projection_collection']
        self.model_collection = self.data_block[section_names.model,'collection']
        self.data_vector = None
        if self.data_load:
            self.data_vector = syntax.load_auto(self.data_load,data_block=self.data_block,default_section=section_names.data,loader=DataVector.load_auto,squeeze=True)

    def execute(self):
        self.style.mpicomm = self.mpicomm
        if self.xmodel is not None:
            x = []
            xmodels, projs = self.xmodel, self.projection_collection.projs
            if isinstance(xmodels,dict) or np.ndim(xmodels[0]) == 0:
                xmodels = [xmodels]*len(projs)
            for xmodel,proj in zip(xmodels,projs):
                kwargs = xmodel.copy()
                if isinstance(xmodel,dict):
                    if xmodel.get('min',None) is None:
                        kwargs['min'] = self.data_vector.get_x(proj=proj).min()
                    if xmodel.get('max',None) is None:
                        kwargs['max'] = self.data_vector.get_x(proj=proj).max()
                    x.append(utils.customspace(**kwargs))
                else:
                    x.append(xmodel)
            data_vector = DataVector(x=x,proj=projs)
            self.projection_collection = self.projection_collection.copy()
            self.projection_collection.setup(data_vector=data_vector)
        data_vectors = [self.projection_collection.to_data_vector(self.model_collection)]
        if self.data_vector is not None:
            data_vectors.append(self.data_vector)
        if self.covariance_load:
            self.style.plot(data_vectors,covariance=self.data_block[self.covariance_load],error_mean=0)
        else:
            self.style.plot(data_vectors)
        if self.save_model: data_vectors[0].save_auto(self.save_model)

    def cleanup(self):
        pass
