from cosmopipe.lib import theory
from cosmopipe import section_names


class LinearModel(object):

    def setup(self):
        pklin = self.data_block[section_names.linear_perturbations,'pk_callable']
        self.growth_rate = self.data_block[section_names.background,'growth_rate']
        model = theory.LinearModel(pklin=pklin,FoG=self.options['FoG'],cosmo={'growth_rate':self.growth_rate})
        effectap = theory.EffectAP(pk_mu=model.pk_mu)
        self.model = effectap
        self.data_shotnoise = self.data_block.get(section_names.data,'shotnoise',0.)

    def execute(self):
        sigmav = self.data_block.get(section_names.galaxy_bias,'sigmav',0.)
        b1 = self.data_block.get(section_names.galaxy_bias,'b1',1.)
        shotnoise = self.data_block.get(section_names.galaxy_bias,'As',1.)*self.data_shotnoise
        f = self.data_block.get(section_names.galaxy_rsd,'f',self.growth_rate)
        qpar = self.data_block.get(section_names.ap_effect,'qpar',1.)
        qperp = self.data_block.get(section_names.ap_effect,'qperp',1.)

        def pk_mu_callable(k,mu):
            return self.model.pk_mu(k,mu,qpar=qpar,qperp=qperp,b1=b1,sigmav=sigmav,shotnoise=shotnoise)

        self.data_block[section_names.model,'y_callable'] = self.data_block[section_names.galaxy_power,'pk_mu_callable'] = pk_mu_callable

    def cleanup(self):
        pass
