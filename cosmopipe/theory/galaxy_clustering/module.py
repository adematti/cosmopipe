from cosmopipe.parameters import ParameterizedModule

from cosmopipe import section_names


class PTModule(ParameterizedModule):

    def set_primordial(self):
        self.zeff = self.data_block[section_names.survey_selection,'zeff']
        pklin = self.data_block[section_names.primordial_perturbations,'pk_callable']
        _new_pklin = pklin.tag != self._cache.get('pklin_tag',None) or self.zeff is not self._cache.get('zeff',None)
        self._cache['pklin_tag'] = pklin.tag
        self._cache['zeff'] = self.zeff
        #print('lol',_new_pklin)
        if _new_pklin:
            self.pklin = pklin.to_1d(z=self.zeff)
            self.klin = self.pklin.k
            self.radius_sigma = self.data_block.get(section_names.galaxy_rsd,'radius_sig',8.)
            self.sigma = self.pklin.sigma_r(self.radius_sigma)
        return _new_pklin
