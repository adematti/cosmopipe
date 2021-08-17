import os
from pypescript.main import main
from pypescript.syntax import yaml_parser

from cosmopipe.lib import setup_logging


def test_demo():

    configs = ['demo_basic.yaml','demo_basic_sum.yaml','demo_basic_joint.yaml']
    configs += ['demo_e2e_correlation_function.yaml','demo_e2e_power_spectrum.yaml']
    configs += ['demo_linear_fullfit.yaml','demo_linear_templatefit.yaml']
    configs += ['demo_pybird.yaml','demo_velocileptors.yaml']
    configs += ['demo_mock_covariance.yaml','demo_window_function.yaml']
    configs += ['demo_plotting.yaml','demo_samplers.yaml']
    #configs += ['demo_mock_challenge_templatefit.yaml']

    for config in configs:
        with open(config,'r') as file:
            toret = file.read()
        config_block = yaml_parser(toret)
        try:
            main(config_block=config_block)
        except Exception as exc: 
            raise RuntimeError('Exception in demo {}.'.format(config)) from exc          

 
if __name__ == '__main__':

    setup_logging()
    test_demo()
