from pypescript.main import main
from pypescript.syntax import yaml_parser

from cosmopipe.lib import setup_logging


def test_demo_mock():

    with open('demo_xlim.yaml','r') as file:
        toret = file.read()
    config_block = yaml_parser(toret)
    main(config_block=config_block)


if __name__ == '__main__':

    setup_logging()
    test_demo_mock()
