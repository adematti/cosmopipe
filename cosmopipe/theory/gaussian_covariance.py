import numpy as np
from scipy import special
from pypescript import SectionBlock

from cosmopipe import section_names
from cosmopipe.lib import syntax, utils
from cosmopipe.lib.theory import gaussian_covariance
from cosmopipe.lib.theory.integration import MultipoleExpansion
from cosmopipe.lib.theory.projection import ProjectionBase
from cosmopipe.data.data_vector import get_kwview


class GaussianPkCovariance(object):

    def setup(self):
        kwargs = {}
        for name in ['edges','k','projs','volume','shotnoise','integration']:
            if name in self.options:
                kwargs[name] = self.options[name]
        if 'edges' in kwargs and isinstance(kwargs['edges'],dict):
            kwargs['edges'] = utils.customspace(**kwargs['edges'])
        self.kwview = {}
        data_load = self.options.get('data_load',kwargs.get('edges',None) is None)
        if data_load:
            if isinstance(data_load,bool) and data_load:
                data_load = 'data_vector'
            self.data_vector = self.data_block[syntax.split_sections(data_load,default_section=section_names.data)]
            self.kwview = self.data_vector.kwview
            dkwargs = {'projs':self.data_vector.get_projs()}
            for name in ['edges','volume','shotnoise']:
                dkwargs[name] = self.data_vector.attrs.get(name,None)
            utils.dict_nonedefault(kwargs,**dkwargs)
            utils.dict_nonedefault(kwargs,k=self.data_vector.noview().get_x(proj=kwargs['projs'],concatenate=False))
        kwargs['edges'] = np.array(kwargs['edges'])
        self.model_base = self.data_block[section_names.model,'y_base']
        if self.model_base.mode == ProjectionBase.MULTIPOLE:
            self.multipole_expansion = MultipoleExpansion(ells=self.model_base.projs)
        self.covariance = gaussian_covariance.GaussianPkCovarianceMatrix(**kwargs)

    def execute(self):
        y_callable = self.data_block[section_names.model,'y_callable']
        if self.model_base.mode == ProjectionBase.MULTIPOLE:
            self.multipole_expansion.input_fun = y_callable
            y_callable = self.multipole_expansion

        self.covariance.compute(y_callable)
        xlim = self.options.get('xlim',None)
        if xlim is not None:
            kwview = get_kwview(self.covariance.x[0],xlim=xlim)
        else:
            kwview = self.kwview
        cov = self.covariance.view(**kwview)
        self.data_block[section_names.covariance,'covariance_matrix'] = cov
        self.data_block[section_names.covariance,'cov'] = cov.get_cov()
        self.data_block[section_names.covariance,'invcov'] = cov.get_invcov()
        self.data_block[section_names.covariance,'nobs'] = None

    def cleanup(self):
        pass
