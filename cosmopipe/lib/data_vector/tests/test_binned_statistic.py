import numpy as np

from cosmopipe.lib import utils, setup_logging
from cosmopipe.lib.data_vector import BinnedStatistic, BinnedProjection, ProjectionName


def test_binned_statistic():
    edges = np.linspace(0.,1.,10)
    k = (edges[:-1] + edges[1:])/2.
    test = BinnedStatistic(data={'k':k},edges={'k':edges})
    test.set_new_edges({'k':edges})
    assert test['k'].size == edges.size - 1
    edges = np.linspace(0.1,0.9,5)
    test.set_new_edges({'k':edges})
    assert test['k'].size == edges.size - 1


def test_binned_projection():
    edges = np.linspace(0.,1.,10)
    k = (edges[:-1] + edges[1:])/2.
    test = BinnedProjection(data={'k':k},x='k',edges={'k':edges})
    assert np.allclose(test.get_x(),k)
    mask = k < 0.5
    assert np.all(test.get_edges(mask=mask)[0] < 0.5)
    assert np.all(test.get_edges(mask=np.flatnonzero(mask))[0] < 0.5)


def test_projection():
    ell0 = ProjectionName('ell_0')
    assert ell0.mode == 'multipole' and ell0.proj == 0
    mu0 =  ProjectionName('mu_0_0.5')
    assert mu0.mode == 'muwedge' and mu0.proj == (0.,0.5)
    mu0 = ProjectionName('mu_0_1/3')
    assert mu0.mode == 'muwedge' and mu0.proj == (0.,1./3)
    assert ProjectionName(None).mode == None
    ell0 = ProjectionName('corr_ell_0')
    assert ell0.space == 'correlation'


if __name__ == '__main__':

    setup_logging()
    #test_binned_statistic()
    #test_binned_projection()
    test_projection()
