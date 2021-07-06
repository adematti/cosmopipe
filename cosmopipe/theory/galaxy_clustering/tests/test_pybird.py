import numpy as np
from pypescript import mimport

from cosmopipe import section_names
from cosmopipe.lib import setup_logging


def test_pybird():

    primordial = mimport('cosmopipe.theory.primordial.primordial',options={'engine':'eisenstein_hu','compute':'pk_m'})
    primordial.setup()
    primordial.execute()
    data_block = primordial.data_block
    pybird = mimport('cosmopipe.theory.galaxy_clustering.pybird',data_block=data_block,options={'data_shotnoise':1000.})
    data_block[section_names.survey_geometry,'zeff'] = 1.
    pybird.setup()
    for name,value in zip(['b1', 'b2', 'b3', 'b4', 'cct', 'cr1','cr2'],[2., 0.8, 0.2, 0.8, 0.2, -4., 0]):
        data_block[section_names.galaxy_bias,name] = value
    pybird.execute()
    k = np.linspace(0.01,0.2)
    pk1 = data_block[section_names.model,'collection'].get_by_proj()(k)
    #data_block[section_names.galaxy_bias,'b1'] += 1.
    data_block[section_names.galaxy_rsd,'fsig'] = 0.5
    pybird.execute()
    pk2 = data_block[section_names.model,'collection'].get_by_proj()(k)
    assert not np.all(pk1 == pk2)
    try:
        import timeit
    except ImportError:
        return

    d = {}
    d['pybird execute'] = {'stmt':"pybird.execute()",'number':10}
    d['pybird full'] = {'stmt':"pybird.setup(); pybird.execute()",'number':10}
    for key,value in d.items():
        dt = timeit.timeit(**value,globals={**globals(),**locals()})/value['number']*1e3
        print('{} takes {:.3f} milliseconds'.format(key,dt))


if __name__ == '__main__':

    setup_logging()
    test_pybird()
