import os
import numpy as np

from cosmopipe.lib.theory import PkEHNoWiggle, LinearModel, GaussianPkCovarianceMatrix
from cosmopipe.lib.data import MockDataVector
from cosmopipe.lib.data.plotting import PowerSpectrumPlotStyle, CovarianceMatrixPlotStyle
from cosmopipe.lib import setup_logging

base_dir = os.path.dirname(os.path.realpath(__file__))
plot_dir = os.path.join(base_dir,'plots')


def test_pk_covariance():

    pk = PkEHNoWiggle(k=np.linspace(0.001,0.5,100))
    pk.run(sigma8=1.)
    model = LinearModel(pklin=pk,cosmo={'growth_rate':0.8})
    kedges = np.linspace(0.01,0.4,40)
    cov = GaussianPkCovarianceMatrix(kedges,projs=('ell_0','ell_2','ell_4'),volume=(1e3)**3,shotnoise=1e3)
    cov.run(pk_mu=model.pk_mu)
    print(cov.x[0].projs)
    filename = os.path.join(plot_dir,'covariance.png')
    cov.plot(filename=filename,style='corr',data_styles='pk')
    filename = os.path.join(plot_dir,'mean.png')
    cov.x[0].plot(filename=filename,style='pk')

    data = MockDataVector(cov)
    filename = os.path.join(plot_dir,'real.png')
    data.plot(filename=filename,style='pk')



if __name__ == '__main__':

    setup_logging()
    test_pk_covariance()
