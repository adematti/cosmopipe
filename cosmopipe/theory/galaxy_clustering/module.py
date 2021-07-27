from cosmopipe.parameters import ParameterizedModule

from cosmopipe import section_names


class PTModule(ParameterizedModule):

    def set_parameters(self, *args, **kwargs):
        super(PTModule,self).set_parameters(*args,**kwargs)
        self.derive_fsig = (section_names.galaxy_rsd,'fsig') in self._derived_parameters

    def set_primordial(self):
        self.zeff = self.data_block[section_names.survey_selection,'zeff']
        pklin = self.data_block[section_names.primordial_perturbations,'pk_callable']
        _new_pklin = pklin.interp is not self._cache.get('pklin_eval',None) or self.zeff is not self._cache.get('zeff',None)
        self._cache['pklin_eval'] = pklin.interp
        self._cache['zeff'] = self.zeff
        if _new_pklin:
            self.pklin = pklin.to_1d(z=self.zeff)
            self.klin = self.pklin.k
            self.sigma8 = self.pklin.sigma8()
            fo = self.data_block[section_names.primordial_cosmology,'cosmo'].get_fourier()
            self.growth_rate = fo.sigma8_z(self.zeff,of='theta_cb')/fo.sigma8_z(self.zeff,of='delta_cb')
        return _new_pklin
