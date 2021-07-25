from cosmopipe import section_names
from cosmopipe.lib.survey_selection import binning


class BaseBinning(object):

    def setup(self):
        options = dict()
        matrix = binning.BaseBinning(**options)
        self.data_block[section_names.survey_selection,'operations'] = self.data_block.get(section_names.survey_selection,'operations',[]) + [matrix]

    def execute(self):
        pass

    def cleanup(self):
        pass
