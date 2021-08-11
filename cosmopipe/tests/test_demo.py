from pypescript.main import main
from pypescript.syntax import yaml_parser

from cosmopipe.lib import setup_logging


def test_demo():

    configs = ['demo_basic.yaml','demo_linear_templatefit.yaml','demo_plotting.yaml','demo_samplers.yaml']
    #configs += ['demo_mock_challenge_templatefit.yaml']

    for config in configs:
        with open(config,'r') as file:
            toret = file.read()
        config_block = yaml_parser(toret)
        main(config_block=config_block)


if __name__ == '__main__':

    setup_logging()
    test_demo()
