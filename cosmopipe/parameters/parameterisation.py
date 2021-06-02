import logging

import numpy as np
from pypescript.config import ConfigError
from pypescript import syntax

from cosmopipe import section_names
from cosmopipe.lib.parameter import ParamBlock, ParamError


class Parameterisation(object):

    #_reserved_option_names = ['common_parameters','specific_parameters']
    logger = logging.getLogger('Parameterisation')

    def setup(self):
        self.parameters = ParamBlock(syntax.collapse_sections(self.options.get_dict('common_parameters',{}),maxdepth=2))
        specific_modules = syntax.collapse_sections(self.options.get_dict('specific_parameters',{}),maxdepth=2)
        for module_name,specific in specific_modules.items():
            if module_name not in self.config_block:
                raise ConfigError('Specific parameters provided for module [{}] which does not exist.'.format(module_name))
            mapping = {}
            specific = ParamBlock(specific)
            for param in specific:
                param_name = param.name
                param.add_suffix(module_name)
                self.log_info('Setting specific parameter {} for module [{}] as {}.'.format(param_name,module_name,param.name),rank=0)
                if param in self.parameters:
                    raise ParamError('Attempting to rename specific parameter {} for module [{}] as {},\
                                    which already exists in common_parameters'.format(param_name,module_name,param.name))
                mapping[param_name.tuple] = param.name.tuple
            self.config_block[module_name,syntax.datablock_mapping].update(mapping)
            self.parameters.update(specific)
        for param in self.parameters:
            if param.value is None:
                raise ParamError('An initial value must be provided for parameter {}.'.format(param.name))
        for param in self.parameters:
            self.data_block[param.name.tuple] = param.value
        self.data_block[section_names.parameters,'list'] = self.parameters

    def execute(self):
        pass

    def cleanup(self):
        pass
