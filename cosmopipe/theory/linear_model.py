from cosmopipe.lib import theory
from cosmopipe import section_names


class LinearModel(object):

    def setup(self):
        pklin = self.data_block[section_names.linear_perturbations,'pk_callable']
        self.sigma8 = pklin.sigma8()
        self.growth_rate = self.data_block[section_names.background,'growth_rate']
        model = theory.LinearModel(pklin=pklin,FoG=self.options['FoG'],cosmo={'f':self.growth_rate})
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

        def pk_mu_callable(k,mu):
            return self.model.pk_mu(k,mu,b1=b1,sigmav=sigmav,shotnoise=shotnoise,f=fsig/self.sigma8)

        self.data_block[section_names.model,'y_callable'] = self.data_block[section_names.galaxy_power,'pk_mu_callable'] = pk_mu_callable

    def cleanup(self):
        pass
