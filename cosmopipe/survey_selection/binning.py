from cosmopipe import section_names
from cosmopipe.lib.survey_selection import binning


class BaseBinning(object):

    def setup(self):
        options = dict()
        self.matrix = binning.BaseBinning(**options)
        self.data_block[section_names.survey_selection,'effect'] = self.matrix

    def execute(self):
        pass

    def cleanup(self):
        pass
