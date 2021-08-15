from cosmopipe import section_names
from cosmopipe.data_vector.data_vector import get_data_from_options

from cosmopipe.lib.theory import ProjectionBasis
from cosmopipe.lib.data_vector import ProjectionName
from cosmopipe.lib.survey_selection import WindowFunction, PowerWindowMatrix
from cosmopipe.lib.survey_selection.window_function import compute_real_window_1d


class WindowConvolution(object):

    def setup(self):
        print(self.options)
        options = dict(krange=None,srange=None,ns=1024,q=0,default_zero=False)
        for name,value in options.items():
            options[name] = self.options.get(name,value)

        window = get_data_from_options(self.options,data_load=self.options['window_load'],data_block=self.data_block,default_section=section_names.survey_selection,loader=WindowFunction.load_auto)
        for proj in window.get_projs():
            proj.setdefault(space=ProjectionName.CORRELATION)

        if options['srange'] is None:
            projs = window.get_projs().select(space=ProjectionName.CORRELATION,mode=ProjectionName.MULTIPOLE)
            x = [window.get_x(proj=proj) for proj in projs]
            options['srange'] = (max(x_.min() for x_ in x),min(x_.max() for x_ in x))

        if options['krange'] is None:
            # try get model base
            model_bases = self.data_block[section_names.model,'collection'].bases()
            k = model_bases.select(space=ProjectionBasis.POWER)[0].x
            options['krange'] = (k.min(),k.max())

        matrix = PowerWindowMatrix(window=window,**options)
        # Turn to configuration space if not already
        window_real = compute_real_window_1d(matrix.s,window)
        projs = window.projs
        for proj in window_real.projs:
            if not projs.select(space=ProjectionName.CORRELATION,mode=ProjectionName.MULTIPOLE,proj=proj.proj,wa_order=proj.wa_order):
                window.set(window_real.get(proj))
        self.data_block[section_names.survey_selection,'operations'] = self.data_block.get(section_names.survey_selection,'operations',[]) + [matrix]

    def execute(self):
        pass

    def cleanup(self):
        pass
