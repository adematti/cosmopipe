import os

from pypescript.main import main
from cosmopipe.lib import setup_logging


def test_demo_mock():
    #from pypescript import BasePipeline, ConfigBlock
    #block = ConfigBlock('demo_mock.yaml')
    #print(block)
    #pipeline = BasePipeline(config_block='demo_mock.yaml')
    #pipeline.plot_inheritance_graph(filename='plots/inheritance.ps')
    #pipeline.setup()
    #pipeline.execute()
    #pipeline.cleanup()
    #main(config='demo_mock3_cobaya.yaml')
    #main(config='demo_mock3_cosmosis.yaml')
    #main(config='demo_velocileptors_minuit.yaml')
    #main(config='demo_pybird_minuit.yaml')
    #main(config='demo_xlim_cobaya.yaml')
    #main(config='demo_bao_templatefit_cobaya.yaml')
    pass


test()

if __name__ == '__main__':

    setup_logging()
    test_demo_mock()
