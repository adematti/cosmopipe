import numpy as np

from cosmopipe.lib.theory import PkEHNoWiggle, LinearModel
from cosmopipe.lib import setup_logging


def test_linear():

    pk = PkEHNoWiggle(k=np.linspace(0.01,0.5,100))
    pk.run(sigma8=1.)
    assert np.allclose(pk.sigma8(),1.)
    model = LinearModel(pklin=pk,cosmo={'growth_rate':0.8})
    assert np.allclose(model.pk_mu(k=pk.k,mu=0.),pk['pk'])


if __name__ == '__main__':

    setup_logging()
    test_linear()
