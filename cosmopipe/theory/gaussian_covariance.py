import numpy as np

from pypescript import SectionBlock

from cosmopipe.lib.theory import gaussian_covariance
from cosmopipe import section_names
from cosmopipe.data.data_vector import get_kwview


class GaussianPkCovariance(object):

    def setup(self):
        kwargs = {}
        for arg in gaussian_covariance.GaussianPkCovarianceMatrix.__init__.__code__.co_varnames[1:]:
            if arg in self.options:
                kwargs[arg] = self.options[arg]
        if 'kedges' in kwargs and isinstance(kwargs['kedges'],dict):
            kedges = kwargs['kedges']
            linear = np.arange(kedges['min'],kedges['max'],kedges['step'])
            if kedges.get('binning','linear') == 'log':
                kwargs['kedges'] = 10**linear
            else:
                kwargs['kedges'] = linear
        self.kwview = {}
        if self.options.get('use_data_vector',False):
            self.data_vector = self.data_block.get(section_names.data,'data_vector',None)
            self.kwview = self.data_vector.kwview
            kwargs.setdefault('kedges',self.data_vector.attrs.get('kedges',None))
            kwargs.setdefault('projs',self.data_vector.projs)
            kwargs.setdefault('k',self.data_vector.noview().get_x(proj=kwargs['projs'][0]))
            kwargs.setdefault('volume',self.data_vector.attrs.get('volume',None))
            kwargs.setdefault('shotnoise',self.data_vector.attrs.get('shotnoise',None))
        self.covariance = gaussian_covariance.GaussianPkCovarianceMatrix(**kwargs)

    def execute(self):
        self.covariance.run(self.data_block[section_names.model,'y_callable'])
        if self.options.has('use_data_xlim'):
            self.kwview = get_kwview(self.covariance.x[0],SectionBlock(self.config_block,self.options.get_string('use_data_xlim')))
        cov = self.covariance.view(**self.kwview)
        self.data_block[section_names.covariance,'covariance_matrix'] = cov
        self.data_block[section_names.covariance,'cov'] = cov.get_cov()
        self.data_block[section_names.covariance,'invcov'] = cov.get_invcov()
        self.data_block[section_names.covariance,'nobs'] = None

    def cleanup(self):
        pass
