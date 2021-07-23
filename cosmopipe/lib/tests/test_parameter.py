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
    assert param.prior.dist == 'uniform'
    param.update(prior={'dist':'norm','scale':1.,'loc':0.})
    assert param.prior.dist == 'norm'
    assert param.name == 'parameters.a'


def decode_name(name, size=None):

    import re

    replaces = re.finditer('\[(-?\d*):(\d*):*(-?\d*)\]',name)
    strings, ranges = [], []
    string_start = 0
    for ireplace,replace in enumerate(replaces):
        start, stop, step = replace.groups()
        if not start: start = 0
        else: start = int(start)
        if not stop:
            stop = size
            if size is None:
                raise ValueError('You should provide an upper limit to parameter index')
        else: stop = int(stop)
        if not step: step = 1
        else: step = int(step)
        strings.append(name[string_start:replace.start()])
        string_start = replace.end()
        ranges.append(range(start,stop,step))

    strings += [name[string_start:]]

    return strings,ranges


def yield_names(name, latex=None, size=1):

    strings,ranges = decode_name(name,size=size)

    if not ranges:
        yield strings[0]

    else:
        import itertools

        template = '{:d}'.join(strings)
        if latex is not None:
            latex = latex.replace('[]','{{{:d}}}')
            for nums in itertools.product(*ranges):
                yield template.format(*nums), latex.format(*nums)
        else:
            for nums in itertools.product(*ranges):
                yield template.format(*nums)


def find_names(allnames, name):

    import re

    strings,ranges = decode_name(name,size=1000)
    if not ranges:
        if strings[0] in allnames:
            return [strings[0]]
        return []
    pattern = re.compile('(-?\d*)'.join(strings))
    toret = []
    for paramname in allnames:
        match = re.match(pattern,paramname)
        if match:
            add = True
            for s,ra in zip(match.groups(),ranges):
                idx = int(s)
                add = idx in ra # ra not in memory
                if not add: break
            if add:
                toret.append(paramname)
    return toret


def test_sugar():

    name = 'test'
    toret = []
    for name in yield_names(name, size=1):
        toret.append(name)
    assert toret == [name]

    name = 'test_[-1:3]_[:2]_test'
    toret = []
    for name in yield_names(name, latex='t_[]_[]', size=1):
        toret.append(name[0])
    assert toret == ['test_-1_0_test', 'test_-1_1_test', 'test_0_0_test', 'test_0_1_test',
                    'test_1_0_test', 'test_1_1_test', 'test_2_0_test', 'test_2_1_test']

    name = 'test_[-1:3]_[:2]_test'
    allnames = ['test_1_1_test','test_2_1_test','ok']
    assert find_names(allnames,name) == allnames[:-1]


if __name__ == '__main__':

    setup_logging()
    """
    test_paramname()
    test_prior()
    test_parameter()
    """
    test_sugar()
