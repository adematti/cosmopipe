import numpy as np

from cosmopipe.lib.primordial import Cosmology
from cosmopipe.lib.theory import LinearModel, DataVectorProjection
from cosmopipe.lib import setup_logging


def test_projection():

    pk = Cosmology().get_fourier('eisenstein_hu').pk_interpolator().to_1d()
    model = LinearModel(pklin=pk)
    k = np.linspace(0.02,0.2,15)
    projection = DataVectorProjection(xdata=k,projdata=('ell_0','ell_2'))
    assert len(projection.evalmesh) == 1 and projection.evalmesh[0][0].size == k.size
    pk_ell = projection(model.pk_mu)
    #print(pk_ell)
    assert pk_ell.size == k.size*2
    projection = DataVectorProjection(xdata=[k,k[:10]],projdata=('ell_0','ell_2'))
    assert len(projection.evalmesh) == 1 and projection.evalmesh[0][0].size == k.size
    assert projection(model.pk_mu).size == k.size + 10
    projection = DataVectorProjection(xdata=[k,k[:10]+0.042],projdata=('ell_0','ell_2'))
    assert len(projection.evalmesh) == 1 and projection.evalmesh[0][0].size == k.size + 10
    pk_ell = projection(model.pk_mu)
    assert pk_ell.size == k.size + 10

if __name__ == '__main__':

    setup_logging()
    test_projection()
