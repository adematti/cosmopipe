import numpy as np

from cosmopipe import section_names
from cosmopipe.lib.samples import Samples, Profiles, SamplesPlotStyle, ProfilesPlotStyle
from cosmopipe.lib import syntax, utils


class SamplesPlotting(object):

    def setup(self):
        self.save_stats = self.options.get('save_stats',None)
        if self.save_stats is not None and not isinstance(self.save_stats,list):
            self.save_stats = [self.save_stats]
        self.tablefmt = self.options.get('tablefmt','latex_raw')
        self.toplot = self.options['toplot']
        self.burnin = self.options.get('burnin',None)
        self.samples_load = self.options.get('samples_load','samples')
        self.style = SamplesPlotStyle(**syntax.remove_keywords(self.options))

    def execute(self):
        self.style.mpicomm = self.mpicomm
        chains = syntax.load_auto(self.samples_load,data_block=self.data_block,default_section=section_names.likelihood,loader=Samples.load_auto)
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
        self.save_stats = self.options.get('save_stats',None)
        if self.save_stats is not None and not isinstance(self.save_stats,list):
            self.save_stats = [self.save_stats]
        self.tablefmt = self.options.get('tablefmt','latex_raw')
        self.toplot = self.options['toplot']
        self.profiles_load = self.options.get('profiles_load','profiles')
        self.style = ProfilesPlotStyle(**self.options)

    def execute(self):
        self.style.mpicomm = self.mpicomm
        self.style.profiles = profiles = syntax.load_auto(self.profiles_load,data_block=self.data_block,default_section=section_names.likelihood,loader=Profiles.load_auto)
        if self.save_stats:
            for prof,filename in zip(profiles,self.save_stats):
                prof.to_stats(tablefmt=self.tablefmt,filename=filename)
        self.style.profiles = profiles
        for toplot in self.toplot:
            for key,value in toplot.items():
                getattr(self.style,key)(**value)

    def cleanup(self):
        pass
