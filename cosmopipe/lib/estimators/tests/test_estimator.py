import os

import numpy as np

from cosmopipe.lib import setup_logging
from cosmopipe.lib.data_vector import BinnedProjection, DataVector
from cosmopipe.lib.estimators.correlation_function import PairCount, NaturalEstimator, LandySzalayEstimator

base_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(base_dir,'_data')


def test_natural_estimator():
    shape = (50,100)
    D1D2 = R1R2 = PairCount(np.ones(shape),np.prod(shape))
    edges = {'s':np.linspace(10.,200.,shape[0]+1),'mu':np.linspace(0.,1.,shape[1]+1)}
    def mid(array): return (array[:-1] + array[1:])/2.
    s,mu = np.meshgrid(mid(edges['s']),mid(edges['mu']),indexing='ij')
    estimator = NaturalEstimator(D1D2,R1R2,data={'s':s,'mu':mu},edges=edges)
    poles = estimator.project_to_multipoles()
    assert np.allclose(poles[0].get_x(),s[:,0])
    wedges = estimator.project_to_muwedges(3)
    assert np.allclose(wedges[0].get_x(),s[:,0])
    wedges = estimator.project_to_muwedges([(0.,1./3.),(1./3.,2./3.)])
    assert np.allclose(wedges[0].get_x(),s[:,0])
    data_fn = os.path.join(data_dir,'data.txt')
    estimator.save_txt(data_fn)
    estimator = BinnedProjection.load_txt(data_fn)
    assert estimator.__class__.__name__ == 'NaturalEstimator'
    assert estimator.shape == shape


def test_landy_szalay_estimator():
    shape = (50,100)
    D1D2 = D1R2 = R1R2 = PairCount(np.ones(shape),np.prod(shape))
    edges = {'s':np.linspace(10.,200.,shape[0]+1),'mu':np.linspace(0.,1.,shape[1]+1)}
    def mid(array): return (array[:-1] + array[1:])/2.
    s,mu = np.meshgrid(mid(edges['s']),mid(edges['mu']),indexing='ij')
    estimator = LandySzalayEstimator(D1D2,R1R2,D1R2=D1R2,data={'s':s,'mu':mu},edges=edges)
    poles = estimator.project_to_multipoles()
    assert np.allclose(poles[0].get_x(),s[:,0])
    wedges = estimator.project_to_muwedges(3)
    assert np.allclose(wedges[0].get_x(),s[:,0])
    wedges = estimator.project_to_muwedges([(0.,1./3.),(1./3.,2./3.)])
    assert np.allclose(wedges[0].get_x(),s[:,0])
    data_fn = os.path.join(data_dir,'data.txt')
    estimator.save_txt(data_fn)
    estimator = BinnedProjection.load_txt(data_fn)
    assert estimator.__class__.__name__ == 'LandySzalayEstimator'
    assert estimator.shape == shape
    estimator = DataVector.load_txt(data_fn)
    assert estimator.__class__.__name__ == 'DataVector'


if __name__ == '__main__':

    setup_logging()
    test_natural_estimator()
    test_landy_szalay_estimator()
