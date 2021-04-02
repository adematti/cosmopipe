import os

from pypescript.main import main
from cobaya.yaml import yaml_load_file
from cobaya.run import run

from cosmopipe.lib import setup_logging


def test_internal():
    main(config='demo3_cobaya.yaml')


def test_external():

    info = yaml_load_file('./external_cobaya.yaml')
    #from cobaya.model import get_model
    #model = get_model(info)
    #print(model.parameterization)
    updated_info, sampler = run(info)
    assert 'parameters.a' in updated_info['params']
    assert 'sample' in sampler.products()


if __name__ == '__main__':

    setup_logging()
    test_internal()
    #test_external()
