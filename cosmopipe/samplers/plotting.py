import numpy as np
from pypescript import syntax

from cosmopipe import section_names
from cosmopipe.lib.samples import Samples, Profiles, SamplesPlotStyle, ProfilesPlotStyle
from cosmopipe.lib import utils


class SamplesPlotting(object):

    def setup(self):
        self.save_stats = self.options.get('save_stats',False)
        self.tablefmt = self.options.get('tablefmt','latex_raw')
        self.toplot = self.options['toplot']
        self.burnin = self.options.get('burnin',None)
        self.samples_load = self.options.get('samples_load',[])
        samples_keys = self.options.get('samples_keys','samples' if not self.samples_load else [])
        if np.isscalar(self.save_stats):
            self.save_stats = [self.save_stats]
        if np.isscalar(self.samples_load):
            self.samples_load = [self.samples_load]
        if np.isscalar(samples_keys):
            samples_keys = [samples_keys]
        self.samples_keys = []
        for key in samples_keys:
            key = syntax.split_sections(key,default_section=section_names.likelihood)
            self.samples_keys.append(key)
        self.style = SamplesPlotStyle(**syntax.remove_keywords(self.options))

    def execute(self):
        chains = []
        for key in self.samples_keys:
            chains.append(self.data_block[key])
        for fn in self.samples_load:
            chains.append(Samples.load_auto(fn))
        if self.burnin is not None:
            chains = [samples.remove_burnin(self.burnin) for samples in chains]
        #for samples in chains: samples.mpi_scatter()
        self.style.chains = chains
        if self.save_stats:
            for samples,filename in zip(chains,self.save_stats):
                samples.to_stats(tablefmt=self.tablefmt,filename=filename)
        for toplot in self.toplot:
            for key,value in toplot.items():
                getattr(self.style,key)(**value)

    def cleanup(self):
        pass


class ProfilesPlotting(object):

    def setup(self):
        self.save_stats = self.options.get('save_stats',False)
        self.tablefmt = self.options.get('tablefmt','latex_raw')
        self.toplot = self.options['toplot']
        self.profiles_load = self.options.get('profiles_load',[])
        profiles_keys = self.options.get('profiles_keys','profiles' if not self.profiles_load else [])
        if np.isscalar(self.save_stats):
            self.save_stats = [self.save_stats]
        if np.isscalar(self.profiles_load):
            self.profiles_load = [self.profiles_load]
        if np.isscalar(profiles_keys):
            profiles_keys = [profiles_keys]
        self.profiles_keys = []
        for key in profiles_keys:
            key = syntax.split_sections(key,default_section=section_names.likelihood)
            self.profiles_keys.append(key)
        self.style = ProfilesPlotStyle(**self.options)

    def execute(self):
        profiles = []
        for key in self.profiles_keys:
            profiles.append(self.data_block[key])
        for fn in self.profiles_load:
            profiles.append(Profiles.load_auto(fn))
        self.style.profiles = profiles
        if self.save_stats:
            for prof,filename in zip(profiles,self.save_stats):
                prof.to_stats(tablefmt=self.tablefmt,filename=filename)
        self.style.profiles = profiles
        for toplot in self.toplot:
            for key,value in toplot.items():
                getattr(self.style,key)(**value)

    def cleanup(self):
        pass
