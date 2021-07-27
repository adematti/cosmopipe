from cosmopipe import section_names
from cosmopipe.data_vector.data_vector import get_data_from_options

from cosmopipe.lib.theory.base import ProjectionBase
from cosmopipe.lib.survey_selection import WindowFunction, PowerWindowMatrix


class WindowConvolution(object):

    def setup(self):
        options = dict(krange=None,srange=None,ns=1024,q=1.5)
        for name,value in options.items():
            options[name] = self.options.get(name,value)

        window = get_data_from_options(self.options,data_load=self.options['window_load'],data_block=self.data_block,default_section=section_names.survey_selection,loader=WindowFunction.load_auto)

        if options['srange'] is None:
            x = window.get_x()
            options['srange'] = (max(x_.min() for x_ in x),min(x_.max() for x_ in x))

        if options['krange'] is None:
            # try get model base
            model_bases = self.data_block[section_names.model,'collection'].bases()
            k = model_bases.select(space=ProjectionBase.POWER)[0].x
            options['krange'] = (k.min(),k.max())

        matrix = PowerWindowMatrix(window=window,**options)
        self.data_block[section_names.survey_selection,'operations'] = self.data_block.get(section_names.survey_selection,'operations',[]) + [matrix]

    def execute(self):
        pass

    def cleanup(self):
        pass
