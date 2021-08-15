import logging

import numpy as np
from pypescript import BasePipeline

from cosmopipe import section_names
from cosmopipe.lib.survey_selection import ModelProjection, ModelProjectionCollection
from cosmopipe.lib.data_vector import DataVector


class SurveyProjection(BasePipeline):

    logger = logging.getLogger('SurveyProjection')

    def setup(self):
        integration = self.options.get('integration',None)
        projs = self.options.get('projs',None)
        xmodels = self.options.get('x',None)
        self.model_collection = self.data_block[section_names.model,'collection']
        data_vector = self.data_block[section_names.data,'data_vector']
        self.projection_collection = ModelProjectionCollection()
        groups = self.projection_collection.propose_groups(projs)
        if len(groups) > 1:
            raise ValueError('Found non-trivial projections in different spaces: {}. Restrict to specific projections.'.format(groups))

        self.projection = ModelProjection(data_vector=data_vector,projs=projs,model_bases=self.model_collection.bases(),integration=integration)
        self.projection_collection.set(self.projection)

        self.pipe_block = self.data_block.copy()
        self.pipe_block[section_names.survey_selection,'operations'] = []
        current_operation = None
        for todo in self.setup_todos:
            todo()
        for operation in self.pipe_block[section_names.survey_selection,'operations']:
            self.projection.append(operation)

        self.projection_collection.setup()
        self.data_block[section_names.survey_selection,'projection'] = self.projection
        projection_collection = self.data_block.get(section_names.survey_selection,'projection_collection',[])
        projection_collection += self.projection_collection
        self.data_block[section_names.survey_selection,'projection_collection'] = projection_collection

    def execute(self):
        self.data_block[section_names.model,'y'] = self.projection(self.model_collection)

    def cleanup(self):
        pass
