import os

import numpy as np
from scipy import interpolate
from cosmoprimo import Cosmology, PowerToCorrelation

from cosmopipe.lib.theory import LinearModel, DataVectorProjection
from cosmopipe.lib import setup_logging


base_dir = os.path.dirname(os.path.realpath(__file__))
plot_dir = os.path.join(base_dir,'_plots')


def test_projection():

    pk = Cosmology().get_fourier('eisenstein_hu').pk_interpolator().to_1d()
    model = LinearModel(pklin=pk)
    k = np.linspace(0.02,0.2,15)
    projection = DataVectorProjection(x=k,projs=('ell_0','ell_2'))
    assert len(projection.evalmesh) == 1 and projection.evalmesh[0][0].size == k.size
    pk_ell = projection(model.pk_mu)
    #print(pk_ell)
    assert pk_ell.size == k.size*2
    projection = DataVectorProjection(x=[k,k[:10]],projs=('ell_0','ell_2'))
    assert len(projection.evalmesh) == 1 and projection.evalmesh[0][0].size == k.size
    assert projection(model.pk_mu).size == k.size + 10
    projection = DataVectorProjection(x=[k,k[:10]+0.042],projs=('ell_0','ell_2'))
    assert len(projection.evalmesh) == 1 and projection.evalmesh[0][0].size == k.size + 10
    pk_ell = projection(model.pk_mu)
    assert pk_ell.size == k.size + 10


def test_plot():
    pk = Cosmology().get_fourier('eisenstein_hu').pk_interpolator().to_1d()
    model = LinearModel(pklin=pk)

    def pk_mu(*args,**kwargs):
        return model.pk_mu(*args,f=0.5,sigmav=4.,**kwargs)

    k = np.linspace(0.02,0.4,15)
    projection = DataVectorProjection(x=k,projs=('ell_0','ell_2','mu_0_1/3','mu_1/3_2/3'))
    #projection = DataVectorProjection(x=k,projs=('ell_0','ell_2'))
    data_vector = projection.to_data_vector(pk_mu)
    filename = os.path.join(plot_dir,'projection.png')
    data_vector.plot(style='pk',filename=filename)


def test_hankel(plot_fftlog=False):
    pk = Cosmology().get_fourier('eisenstein_hu').pk_interpolator().to_1d()
    model = LinearModel(pklin=pk)
    ells = [0,2,4]
    projs = ['ell_{:d}'.format(ell) for ell in ells]
    projection = DataVectorProjection(x=pk.k,projs=projs,model_base='muwedge')
    pkell = projection(model.pk_mu,concatenate=False)
    fftlog = PowerToCorrelation(pk.k,ell=ells,q=1.5)
    s,xiell = fftlog(pkell)
    s = s[0]
    if plot_fftlog:
        from matplotlib import pyplot as plt
        plt.plot(s,s[:,None]**2*xiell.T)
        plt.xlim(10,300)
        plt.show()
    xi_interp = interpolate.interp1d(s,xiell.T,axis=0,kind='cubic',bounds_error=True,assume_sorted=True)
    x = np.linspace(10,200,30)
    projection = DataVectorProjection(x=x,projs=('ell_0','ell_2','ell_4'),model_base=('multipole',ells))
    data_vector = projection.to_data_vector(xi_interp)
    filename = os.path.join(plot_dir,'xi_projection.png')
    data_vector.plot(style='xi',filename=filename)


if __name__ == '__main__':

    setup_logging()
    #test_projection()
    test_plot()
    #test_hankel()
