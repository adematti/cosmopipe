import os
import pkg_resources

from pypescript.main import main
from cosmosis.runtime.config import Inifile
from cosmosis.runtime.pipeline import LikelihoodPipeline

from cosmopipe.lib import setup_logging


def test_internal():
    main(config='demo3_cosmosis.yaml')


def test_external():

    ini = Inifile('external_cosmosis.ini')
    ini.set('cosmopipe','file',os.path.join(pkg_resources.resource_filename('cosmopipe','samplers'),'cosmosis','likelihood.py'))
    pipeline = LikelihoodPipeline(ini)
    data = pipeline.run_parameters([0.2])
    assert data['likelihoods','cosmopipe_like'] != 0.

    from cosmosis.samplers.emcee.emcee_sampler import EmceeSampler
    from cosmosis.output.in_memory_output import InMemoryOutput
    output = InMemoryOutput()
    sampler = EmceeSampler(ini, pipeline, output)
    sampler.config()

    while not sampler.is_converged():
        sampler.execute()


if __name__ == '__main__':

    setup_logging()
    test_internal()
    test_external()
