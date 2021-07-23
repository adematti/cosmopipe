import logging

import numpy as np
from pypescript import BasePipeline

from cosmopipe import section_names
from cosmopipe.lib.survey_selection import ModelProjectionCollection
from cosmopipe.lib.data_vector import DataVector


class SurveyProjection(BasePipeline):

    logger = logging.getLogger('SurveyProjection')

    def setup(self):
        integration = self.options.get('integration',None)
        projs = self.options.get('projs',None)
        xmodels = self.options.get('x',None)
        model_bases = self.data_block[section_names.model,'collection'].bases
        data = self.data_block[section_names.data,'data_vector']
        self.projection_collection = ModelProjectionCollection(data=data,projs=projs,model_base=model_bases,integration=integration)
        self.projection = self.projection_collection.model_projections[0]

        self.pipe_block = self.data_block.copy()
        current_operation = None
        for todo in self.setup_todos:
            todo()
            operation = self.pipe_block.get(section_names.survey_selection,'effect',None)
            if operation is not current_operation:
                self.projection.append(operation)
                current_operation = operation

        if len(self.projection.operations) != 0:
            if len(self.projection.operations) > 1 or type(self.projection.operations[0]) is not BaseBinning:
                if len(self.projection_collection.model_projections) > 1:
                    raise ValueError('Found non-trivial projections in different spaces. Restrict to specific projections.')

        self.projection.setup()
        self.data_block[section_names.survey_selection,'projection'] = self.projection
        self.data_block[section_names.survey_selection,'projection_collection'] = self.data_block.get(section_names.survey_selection,'projection_collection',[]) + self.projection_collection

    def execute(self):
        collection = self.data_block[section_names.model,'collection']
        self.data_block[section_names.model,'y'] = self.projection(collection)
        self.data_block[section_names.survey_selection,'projection'] = self.projection
        #self.data_block[section_names.survey_selection,'projection_collection'] = self.data_block.get(section_names.survey_selection,'projection_collection',[]) + self.projection_collection

    def cleanup(self):
        pass
