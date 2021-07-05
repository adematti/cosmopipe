from cobaya.likelihood import Likelihood as CobayaLikelihood
from pypescript import BasePipeline

from cosmopipe.lib.primordial import PowerSpectrumInterpolator1D
from cosmopipe import section_names
from cosmopipe.lib import mpi


class CosmopipeLikelihood(CobayaLikelihood):

    config_file = 'config.yaml'
    requirements = []
    #param_file = 'param.yaml'
    renames = {}
    # For classy...
    renames['cosmological_parameters.Omega_cdm'] = 'Omega_cdm'
    renames['cosmological_parameters.omega_cdm'] = 'omega_cdm'
    renames['cosmological_parameters.omega_b'] = 'omega_b'
    renames['cosmological_parameters.Omega_b'] = 'Omega_b'
    renames['cosmological_parameters.h'] = 'h'
    renames['cosmological_parameters.n_s'] = 'n_s'
    renames['cosmological_parameters.sigma8'] = 'sigma8'

    @classmethod
    def cosmopipe_to_cobaya_name(cls, name):
        return cls.renames.get(name,name)

    @classmethod
    def cobaya_to_cosmopipe_name(cls, name):
        if '.' in name:
            return name
        for cosmopipe_name,cobaya_name in cls.renames.items():
            if cobaya_name == name:
                return cosmopipe_name

    def initialize(self):
        self._pipeline = BasePipeline(config_block=self.config_file)
        self._pipeline.setup()
        self._pipeline.data_block = BasePipeline.mpi_distribute(self._pipeline.data_block,dests=self._pipeline.mpicomm.rank,mpicomm=mpi.COMM_SELF)
        from .sampler import get_cobaya_parameter
        self.parameters = self._pipeline.pipe_block[section_names.parameters,'list']
        self.params = {str(param.name):param.value for param in self.parameters}
        #if 'rdrag' in self.get_requirements():
        #    self.params['rdrag'] = {'latex': 'r_\mathrm{drag}'}

    @property
    def set_primordial_perturbations(self):
        return 'primordial_perturbations' in self.requirements

    @property
    def set_background(self):
        return 'background' in self.requirements

    def get_requirements(self):
        toret = {}
        if not self.requirements: return toret
        a = self._pipeline.pipe_block[section_names.background,'scale_factor']
        self.z = 1./a - 1
        if self.set_primordial_perturbations:
            toret['Pk_grid'] = {'z': [self.z], 'k_max': 1.1, 'nonlinear': False, 'vars_pairs': [('delta_tot', 'delta_tot')]}
            toret['Hubble'] = {'z': [0.]}
            toret['fsigma8'] = {'z': [self.z]}
            toret['sigma8'] = {'z': [self.z]}
        if self.set_background:
            toret['Hubble'] = {'z': [0.,self.z]}
            toret['rdrag'] = None
            toret['angular_diameter_distance'] = {'z': [self.z]}
        return toret

    def logp(self, **kwargs):
        self._pipeline.pipe_block = self._pipeline.data_block.copy()
        for param in self.parameters:
            name = str(param.name)
            if name in self.renames and self.requirements:
                self._pipeline.pipe_block[param.name.tuple] = self.theory.get_param(self.cosmopipe_to_cobaya_name(name))
            else:
                self._pipeline.pipe_block[param.name.tuple] = kwargs[name]
        if self.set_primordial_perturbations:
            h = self.theory.get_Hubble(0.,units='km/s/Mpc')/100.
            k, z, pk = self.theory.get_Pk_grid(('delta_tot', 'delta_tot'))
            self._pipeline.pipe_block[section_names.primordial_perturbations,'pk_callable'] = PowerSpectrumInterpolator1D(k=k/h,pk=pk*h**3)
            self._pipeline.pipe_block[section_names.primordial_perturbations,'growth_rate'] = f = self.theory.get_fsigma8(self.z)/self.theory.get_sigma8(self.z)
            self._pipeline.pipe_block[section_names.background,'hubble_rate'] = hubble/h
        if self.set_background:
            h = float(self.theory.get_Hubble(0.,units='km/s/Mpc')/100.)
            hubble = float(self.theory.get_Hubble(self.z,units='km/s/Mpc')/100.)
            self._pipeline.pipe_block[section_names.primordial_perturbations,'sound_horizon_drag'] = float(self.provider.get_param('rdrag'))*h
            self._pipeline.pipe_block[section_names.background,'hubble_rate'] = hubble/h
            self._pipeline.pipe_block[section_names.background,'comoving_angular_distance'] = float(self.theory.get_angular_diameter_distance(self.z))*(1.+self.z)*h
        for todo in self._pipeline.execute_todos:
            todo()
        return self._pipeline.pipe_block[section_names.likelihood,'loglkl']

    def clean(self):
        pass
