from cosmopipe.lib import setup_logging
from cosmopipe.lib.parameter import ParamName, Prior, Parameter


def test_paramname():
    pn = ParamName('parameters.a')
    d = {pn:2}
    assert d[pn] == 2
    assert pn == 'parameters.a'
    assert d['parameters.a'] == 2

    pn = ParamName('a')
    d = {pn:2}
    assert d[pn] == 2
    assert pn == 'a'
    assert d['a'] == 2


def test_prior():
    prior = Prior(dist='uniform')
    assert not prior.is_proper()
    prior = Prior(dist='uniform',limits=(-2,2))
    assert prior.limits == (-2,2)
    prior = Prior(dist='norm',limits=(-2,2),scale=2)
    assert prior.scale == 2


def test_parameter():
    param = Parameter('parameters.a',value=1,prior={'dist':'uniform'},latex='a')


if __name__ == '__main__':

    setup_logging()
    test_paramname()
    test_prior()
    test_parameter()
