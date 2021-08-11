import os

import numpy as np
from scipy import interpolate
from cosmoprimo import Cosmology, PowerToCorrelation

from cosmopipe.lib.theory import LinearModel, ModelEvaluation
from cosmopipe.lib.data_vector import DataVector
from cosmopipe.lib import setup_logging


base_dir = os.path.dirname(os.path.realpath(__file__))
plot_dir = os.path.join(base_dir,'_plots')


def test_evaluation():

    pk = Cosmology().get_fourier('eisenstein_hu').pk_interpolator().to_1d()
    model = LinearModel(pklin=pk)
    k = np.linspace(0.02,0.2,15)
    data = DataVector(x=[k]*2,proj=['ell_0','ell_2'])
    evaluation = ModelEvaluation(data,model_bases={'mode':'muwedge'})
    assert len(list(evaluation.evalmesh.values())[0]) == 1 and list(evaluation.evalmesh.values())[0][0][0].size == k.size
    pk_ell = evaluation(model.pk_mu)
    #print(pk_ell)
    assert pk_ell.size == k.size*2
    data = DataVector(x=[k,k[:10]],proj=['ell_0','ell_2'])
    evaluation = ModelEvaluation(data=data,model_bases={'mode':'muwedge'})
    assert len(list(evaluation.evalmesh.values())[0]) == 1 and list(evaluation.evalmesh.values())[0][0][0].size == k.size
    assert evaluation(model.pk_mu).size == k.size + 10
    data = DataVector(x=[k,k[:10]+0.042],proj=['ell_0','ell_2'])
    evaluation = ModelEvaluation(data,model_bases={'mode':'muwedge'})
    assert len(list(evaluation.evalmesh.values())[0]) == 1 and list(evaluation.evalmesh.values())[0][0][0].size == k.size + 10
    pk_ell = evaluation(model.pk_mu)
    assert pk_ell.size == k.size + 10


def test_plot():
    pk = Cosmology().get_fourier('eisenstein_hu').pk_interpolator().to_1d()
    model = LinearModel(pklin=pk)

    def pk_mu(*args,**kwargs):
        return model(*args,f=0.5,sigmav=4.,**kwargs)

    k = np.linspace(0.02,0.4,15)
    data = DataVector(x=[k]*4,proj=['ell_0','ell_2','mu_0_1/3','mu_1/3_2/3'])
    evaluation = ModelEvaluation(data,model_bases={'mode':'muwedge'})
    #evaluation = ModelEvaluation(x=k,projs=('ell_0','ell_2'))
    data_vector = evaluation.to_data_vector(pk_mu)
    filename = os.path.join(plot_dir,'evaluation.png')
    data_vector.plot(style='power',filename=filename)


def test_hankel(plot_fftlog=True):
    pk = Cosmology().get_fourier('eisenstein_hu').pk_interpolator().to_1d()
    model = LinearModel(pklin=pk)
    ells = [0,2,4]
    projs = ['ell_{:d}'.format(ell) for ell in ells]
    data = DataVector(x=[pk.k]*len(projs),proj=projs)
    evaluation = ModelEvaluation(data,model_bases={'mode':'muwedge'})
    pkell = evaluation(model,concatenate=False)
    pkell = np.array(pkell)
    x = pk.k
    fftlog = PowerToCorrelation(x,ell=ells,q=1.5,minfolds=10)
    damp = np.ones(x.size,dtype='f8')
    cutoff = 2.
    high = x>cutoff
    damp[high] *= np.exp(-1.*(x[high]/cutoff - 1.)**2)
    cutoff = 1e-4
    low = x<cutoff
    damp[low] *= np.exp(-1.*(cutoff/x[low] - 1)**2)
    #print(x.max())
    if plot_fftlog:
        from matplotlib import pyplot as plt
        mask = (x > 0.) & (x < 50)
        plt.loglog(x[mask],pkell.T[mask])
        plt.loglog(x[mask],pkell.T[mask]*damp[mask,None])
        plt.show()
    s,xiell = fftlog(pkell*damp)
    s = s[0]
    if plot_fftlog:
        from matplotlib import pyplot as plt
        mask = (s > 1) & (s < 300)
        plt.plot(s[mask],s[mask,None]**2*xiell.T[mask])
        plt.show()
    xi_interp = interpolate.interp1d(s,xiell.T,axis=0,kind='cubic',bounds_error=True,assume_sorted=True)
    x = np.linspace(10,200,30)
    data = DataVector(x=[x]*len(projs),proj=projs)
    evaluation = ModelEvaluation(data=data,model_bases={'mode':'multipole','projs':ells})
    data_vector = evaluation.to_data_vector(xi_interp)
    filename = os.path.join(plot_dir,'xi_evaluation.png')
    data_vector.plot(style='correlation',filename=filename)


if __name__ == '__main__':

    setup_logging()
    test_evaluation()
    test_plot()
    test_hankel()
