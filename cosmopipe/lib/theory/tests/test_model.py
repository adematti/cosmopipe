import numpy as np

from cosmoprimo import Cosmology

from cosmopipe.lib.theory import LinearModel
from cosmopipe.lib import setup_logging


def test_linear():

    pk = Cosmology().get_fourier(engine='eisenstein_hu').pk_interpolator().to_1d()
    model = LinearModel(pklin=pk)
    assert model(k=pk.k,mu=0.,f=0.8).shape == (pk.k.size,)
    assert model(k=0.1,mu=0.,f=0.8).ndim == 0
    assert model(k=0.1,mu=np.linspace(0.,0.4,3),f=0.8).shape == (3,)
    assert model(k=pk.k,mu=np.linspace(0.,0.4,3),f=0.8).shape == (pk.k.size,3)
    assert np.allclose(model(k=pk.k,mu=0.,f=0.8),pk(pk.k))


if __name__ == '__main__':

    setup_logging()
    test_linear()
