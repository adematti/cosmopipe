from cosmopipe import section_names
from cosmopipe.lib.samples import Samples, SamplesPlotStyle, ProfilesPlotStyle
from cosmopipe.lib import utils


class SamplesPlotting(object):

    def setup(self):
        self.save_stats = self.options.get('save_stats',False)
        self.tablefmt = self.options.get('tablefmt','latex_raw')
        self.toplot = self.options['toplot']
        self.burnin = self.options.get_float('burnin',None)
        self.samples_files = self.options.get('samples_files',[])
        if not isinstance(self.samples_files,list):
            self.samples_files = [self.samples_files]
        samples_keys = self.options.get('samples_keys','samples' if not self.samples_files else [])
        if not isinstance(samples_keys,list):
            samples_keys = [samples_keys]
        self.samples_keys = []
        for key in samples_keys:
            key = utils.split_section_name(key)
            if len(key) == 1:
                key = (section_names.likelihood,) + key
            self.samples_keys.append(key)
        self.style = SamplesPlotStyle(**self.options)

    def execute(self):
        chains = []
        for key in self.samples_keys:
            chains.append(self.data_block[key])
        for fn in self.samples_files:
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
        profiles_keys = self.options.get('profiles_keys',None)
        if profiles_keys is None:
            profiles_keys = ['profiles']
        if not isinstance(profiles_keys,list):
            profiles_keys = eval(profiles_keys)
        self.profiles_keys = []
        for key in profiles_keys:
            key = utils.split_section_name(key)
            if len(key) == 1:
                key = (section_names.likelihood,) + key
            self.profiles_keys.append(key)
        self.style = ProfilesPlotStyle(**self.options)

    def execute(self):
        profiles = []
        for key in self.profiles_keys:
            profiles.append(self.data_block[key])
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
