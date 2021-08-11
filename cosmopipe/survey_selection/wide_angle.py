from cosmopipe import section_names
from cosmopipe.lib.survey_selection import PowerOddWideAngle


class OddWideAngle(object):

    def setup(self):
        options = dict(d=1.,wa_orders=1,los='firstpoint')
        for name,value in options.items():
            options[name] = self.options.get(name,value)

        if options['d'] == 'fiducial':
            zeff = self.data_block.get[section_names.survey_selection,'zeff']
            cosmo = self.data_block.get[section_names.fiducial_cosmology,'cosmo']
            options['d'] = cosmo.get_background().comoving_radial_distance(zeff)

        matrix = PowerOddWideAngle(**options)
        self.data_block[section_names.survey_selection,'operations'] = self.data_block.get(section_names.survey_selection,'operations',[]) + [matrix]

    def execute(self):
        pass

    def cleanup(self):
        pass
