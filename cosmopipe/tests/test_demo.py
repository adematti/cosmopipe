from pypescript.main import main
from cosmopipe.lib import setup_logging


def test_demo_mock():
    main(config_block='demo_xlim.yaml')


if __name__ == '__main__':

    setup_logging()
    test_demo_mock()
