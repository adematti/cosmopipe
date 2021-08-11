from cosmopipe.lib import setup_logging
from cosmopipe.lib.parameter import ParamName, Prior, Parameter, ParameterCollection


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
    assert param.prior.dist == 'uniform'
    param.update(prior={'dist':'norm','scale':1.,'loc':0.})
    assert param.prior.dist == 'norm'
    assert param.name == 'parameters.a'


def test_decode():
    from cosmopipe.lib.parameter import yield_names_latex, find_names_latex, find_names

    name = 'test'
    toret = []
    for name in yield_names_latex(name, default_stop=1):
        toret.append(name)
    assert toret == [name]

    name = 'test_[-1:3]_[:2]_test'
    toret = []
    for name in yield_names_latex(name, latex='t_[]_[]', default_stop=1):
        toret.append(name[0])
    assert toret == ['test_-1_0_test', 'test_-1_1_test', 'test_0_0_test', 'test_0_1_test',
                    'test_1_0_test', 'test_1_1_test', 'test_2_0_test', 'test_2_1_test']

    name = 'test_[-1:3]_[:2]_test'
    allnames = ['test_1_1_test','test_2_1_test','test_2_2_test','ok']
    assert find_names_latex(allnames,name) == [(name,None) for name in allnames[:-2]]

    name = '*.test_[-1:3]_[:2]_test'
    allnames = ['a.test_1_1_test','b.test_2_1_test','b.test_2_2_test','test_2_1_test']
    assert find_names(allnames,name) == allnames[:-2]


    name = ['*.test_[-1:3]_[:2]_test','*.test_2_2_test']
    allnames = ['a.test_1_1_test','b.test_2_1_test','b.test_2_2_test','test_2_1_test']
    assert find_names(allnames,name) == allnames[:-1]

    assert find_names(allnames,['*']) == allnames


def test_collection():

    parameters = ParameterCollection()
    for name in ['a','b','c']:
        param = Parameter('parameters.{}'.format(name),value=1,prior={'dist':'uniform'},latex=name)
        parameters.set(param)


if __name__ == '__main__':

    setup_logging()
    test_paramname()
    test_prior()
    test_parameter()
    test_decode()
    test_collection()
