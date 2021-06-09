from cosmopipe.lib import theory
from cosmopipe import section_names


class LinearModel(object):

    def setup(self):
        zeff = self.data_block[section_names.survey_geometry,'zeff']
        pklin = self.data_block[section_names.primordial_perturbations,'pk_callable'].to_1d(z=zeff)
        self.sigma8 = pklin.sigma8()
        fo = self.data_block[section_names.primordial_cosmology,'cosmo'].get_fourier()
        self.growth_rate = fo.sigma8_z(zeff,of='theta_cb')/fo.sigma8_z(zeff,of='delta_cb')
        model = theory.LinearModel(pklin=pklin,FoG=self.options.get('FoG','gaussian'))
        self.model = theory.EffectAP(pk_mu=model.pk_mu)
        self.data_shotnoise = self.data_block.get(section_names.data,'shotnoise',0.)

    def execute(self):
        sigmav = self.data_block.get(section_names.galaxy_bias,'sigmav',0.)
        b1 = self.data_block.get(section_names.galaxy_bias,'b1',1.)
        shotnoise = self.data_block.get(section_names.galaxy_bias,'As',1.)*self.data_shotnoise
        fsig = self.data_block.get(section_names.galaxy_rsd,'fsig',self.growth_rate*self.sigma8)
        qpar = self.data_block.get(section_names.effect_ap,'qpar',1.)
        qperp = self.data_block.get(section_names.effect_ap,'qperp',1.)
        self.model.set_scaling(qpar=qpar,qperp=qperp)

        def pk_mu_callable(k, mu):
            toret = self.model.pk_mu(k,mu,b1=b1,sigmav=sigmav,shotnoise=shotnoise,f=fsig/self.sigma8)
            return toret

        self.data_block[section_names.model,'y_callable'] = self.data_block[section_names.galaxy_power,'pk_mu_callable'] = pk_mu_callable

    def cleanup(self):
        pass
