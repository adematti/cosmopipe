import logging

import numpy as np
from pypescript.config import ConfigError
from pypescript import syntax

from cosmopipe import section_names
from cosmopipe.lib.modules import ParameterizedModule


class Parameterization(ParameterizedModule):

    logger = logging.getLogger('Parameterization')

    def setup(self):
        self.set_param_block()

    def execute(self):
        pass

    def cleanup(self):
        pass
