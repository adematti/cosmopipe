from cosmopipe.lib import setup_logging
from cosmopipe.lib.parameter import Parameter, ParamName


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


if __name__ == '__main__':

    setup_logging()
    test_paramname()
