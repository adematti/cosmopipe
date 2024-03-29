import logging

import numpy as np
from pypescript.config import ConfigError
from pypescript import syntax

from cosmopipe import section_names

from .parameterized import ParameterizedModule


class Parameterization(ParameterizedModule):

    logger = logging.getLogger('Parameterization')

    def setup(self):
        self.set_parameters()

    def execute(self):
        pass

    def cleanup(self):
        pass
