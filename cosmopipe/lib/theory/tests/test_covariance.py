import os

import numpy as np
from scipy import interpolate

from cosmoprimo import Cosmology

from cosmopipe.lib.theory import LinearModel, GaussianCovarianceMatrix, ModelCollection, ModelEvaluation, ProjectionBase
from cosmopipe.lib.data_vector import DataVector, MockDataVector, BinnedProjection, CovarianceMatrix
from cosmopipe.lib.data_vector.plotting import PowerSpectrumPlotStyle, CovarianceMatrixPlotStyle

from cosmopipe.lib import setup_logging


base_dir = os.path.dirname(os.path.realpath(__file__))
plot_dir = os.path.join(base_dir,'_plots')


def test_pk_multipole_covariance():

    pk = Cosmology().get_fourier('eisenstein_hu').pk_interpolator().to_1d()
    model = LinearModel(klin=pk.k,pklin=pk)
    kedges = np.linspace(0.01,0.4,41)
    k = (kedges[:-1] + kedges[1:])/2.
    proj = [{'space':'power','mode':'multipole','proj':proj} for proj in [0,2,4]]
    data = DataVector(x=k,proj=proj,edges=[{'x':kedges}]*len(proj))
    cov = GaussianCovarianceMatrix(data,model_base=model.base,volume=(1e3)**3)
    #cov = GaussianPkCovarianceMatrix(kedges,projs=('ell_0','ell_2'),volume=(1e3)**3,shotnoise=1e3)
    cov.compute(model)
    filename = os.path.join(plot_dir,'covariance.txt')
    cov.save_auto(filename)
    cov = CovarianceMatrix.load_auto(filename)
    assert cov.cov.shape == (len(k)*len(proj),)*2
    filename = os.path.join(plot_dir,'covariance_multipoles.png')
    cov.plot(filename=filename,style='corr',data_styles='power')
    filename = os.path.join(plot_dir,'mean.png')
    cov.x[0].plot(filename=filename,style='power')

    data = MockDataVector(cov)
    filename = os.path.join(plot_dir,'real.png')
    data.plot(filename=filename,style='power')


def test_pk_muwedge_covariance():

    pk = Cosmology().get_fourier('eisenstein_hu').pk_interpolator().to_1d()
    model = LinearModel(klin=pk.k,pklin=pk)
    kedges = np.linspace(0.01,0.4,41)
    k = (kedges[:-1] + kedges[1:])/2.
    proj = [{'space':'power','mode':'muwedge','proj':proj} for proj in [(0.,1./3.),(1./3.,2./3.),(2./3.,1.)]]
    data = DataVector(x=k,proj=proj,edges=[{'x':kedges}]*len(proj))
    cov = GaussianCovarianceMatrix(data,model_base=model.base,volume=(1e3)**3)
    cov.compute(model)
    assert cov.cov.shape == (len(k)*len(proj),)*2
    filename = os.path.join(plot_dir,'covariance_muwedges.png')
    cov.plot(filename=filename,style='corr',data_styles='power')
    filename = os.path.join(plot_dir,'mean.png')
    cov.x[0].plot(filename=filename,style='power')

    data = MockDataVector(cov)
    filename = os.path.join(plot_dir,'real.png')
    data.plot(filename=filename,style='power')


def test_pk_covariance():

    pk = Cosmology().get_fourier('eisenstein_hu').pk_interpolator().to_1d()
    model = LinearModel(klin=pk.k,pklin=pk)
    kedges = np.linspace(0.01,0.4,41)
    k = (kedges[:-1] + kedges[1:])/2.
    proj = [{'space':'power','mode':'multipole','proj':proj} for proj in [0,2,4]]
    proj += [{'space':'power','mode':'muwedge','proj':proj} for proj in [(0.,1./3.),(1./3.,2./3.),(2./3.,1.)]]
    data = DataVector(x=k,proj=proj,edges=[{'x':kedges}]*len(proj))
    cov = GaussianCovarianceMatrix(data,model_base=model.base,volume=(1e3)**3)
    cov.compute(model)
    assert cov.cov.shape == (len(k)*len(proj),)*2
    filename = os.path.join(plot_dir,'covariance_pk.png')
    cov.plot(filename=filename,style='corr',data_styles='power')
    filename = os.path.join(plot_dir,'mean.png')
    cov.x[0].plot(filename=filename,style='power')

    data = MockDataVector(cov)
    filename = os.path.join(plot_dir,'real.png')
    data.plot(filename=filename,style='power')


def test_xi_covariance():

    pk = Cosmology().get_fourier('eisenstein_hu').pk_interpolator().to_1d()
    model = LinearModel(klin=pk.k,pklin=pk)
    sedges = np.linspace(10.,200,21)
    s = (sedges[:-1] + sedges[1:])/2.
    x, proj, edges = [], [], []
    data = DataVector()
    for ell in [0,2,4]:
        dataproj = BinnedProjection(x=s,proj={'space':'correlation','mode':'multipole','proj':ell},edges={'x':sedges})
        data.set(dataproj)
    cov = GaussianCovarianceMatrix(data,model_base=model.base,volume=(1e3)**3)
    cov.compute(model)
    assert cov.cov.shape == (data.size,)*2
    filename = os.path.join(plot_dir,'covariance_xi.png')
    cov.plot(filename=filename,style='corr')
    filename = os.path.join(plot_dir,'mean.png')
    cov.x[0].plot(filename=filename,style='correlation')

    data = MockDataVector(cov)
    filename = os.path.join(plot_dir,'real.png')
    data.plot(filename=filename,style='correlation')


def test_pk_xi_covariance():

    pk = Cosmology().get_fourier('eisenstein_hu').pk_interpolator().to_1d()
    model = LinearModel(klin=pk.k,pklin=pk)
    kedges = np.linspace(0.01,0.4,41)
    sedges = np.linspace(10.,200,41)
    k = (kedges[:-1] + kedges[1:])/2.
    s = (sedges[:-1] + sedges[1:])/2.
    x, proj, edges = [], [], []
    data = DataVector()
    for ell in [0,2,4]:
        dataproj = BinnedProjection(x=k,proj={'space':'power','mode':'multipole','proj':ell},edges={'x':kedges})
        data.set(dataproj)
    #for ell in [0,2,4]:
    #    dataproj = BinnedProjection(x=s,proj={'space':'correlation','mode':'multipole','proj':ell},edges={'x':sedges})
    #    data.set(dataproj)
    for muwedge in [(0.,1./3.),(1./3.,2./3.),(2./3.,1.)]:
        dataproj = BinnedProjection(x=s,proj={'space':'correlation','mode':'muwedge','proj':muwedge},edges={'x':sedges})
        data.set(dataproj)
    models = ModelCollection([model])
    cov = GaussianCovarianceMatrix(data,model_base=models.bases(),volume=(1e3)**3)
    cov.compute(models)
    assert cov.cov.shape == (data.size,)*2
    filename = os.path.join(plot_dir,'covariance_pk_xi.png')
    cov.plot(filename=filename,style='corr')
    filename = os.path.join(plot_dir,'power_mean.png')
    cov.x[0].plot(filename=filename,style='power')
    filename = os.path.join(plot_dir,'correlation_mean.png')
    cov.x[0].plot(filename=filename,style='correlation')

    data = MockDataVector(cov)
    filename = os.path.join(plot_dir,'power_real.png')
    data.plot(filename=filename,style='power')
    filename = os.path.join(plot_dir,'correlation_real.png')
    data.plot(filename=filename,style='correlation')


def test_pkell_covariance():

    pk = Cosmology().get_fourier('eisenstein_hu').pk_interpolator().to_1d()
    km = pk.k
    model = LinearModel(klin=km,pklin=pk)
    data = DataVector(x=km,proj=['ell_0','ell_2','ell_4'])
    evaluation = ModelEvaluation(data,model_base=model.base)
    pkell = evaluation(model,concatenate=False)
    model = interpolate.interp1d(km,np.array(pkell).T,axis=0,kind='cubic',bounds_error=True,assume_sorted=True)
    model.base = ProjectionBase({'x':km,'space':'power','mode':'multipole','projs':[0,2,4]})

    models = ModelCollection([model])
    kedges = np.linspace(0.01,0.4,41)
    k = (kedges[:-1] + kedges[1:])/2.
    proj = [{'space':'power','mode':'multipole','proj':proj} for proj in [0,2,4]]
    proj += [{'space':'power','mode':'muwedge','proj':proj} for proj in [(0.,1./3.),(1./3.,2./3.),(2./3.,1.)]]
    data = DataVector(x=k,proj=proj,edges=[{'x':kedges}]*len(proj))
    cov = GaussianCovarianceMatrix(data,model_base=models.bases(),volume=(1e3)**3)
    cov.compute(models)
    assert cov.cov.shape == (len(k)*len(proj),)*2
    filename = os.path.join(plot_dir,'covariance_pkell.png')
    cov.plot(filename=filename,style='corr',data_styles='power')
    filename = os.path.join(plot_dir,'mean.png')
    cov.x[0].plot(filename=filename,style='power')

    data = MockDataVector(cov)
    filename = os.path.join(plot_dir,'real.png')
    data.plot(filename=filename,style='power')


def test_view_covariance():

    pk = Cosmology().get_fourier('eisenstein_hu').pk_interpolator().to_1d()
    model = LinearModel(klin=pk.k,pklin=pk)
    kedges = np.linspace(0.01,0.4,41)
    k = (kedges[:-1] + kedges[1:])/2.
    proj = [{'space':'power','mode':'multipole','proj':proj} for proj in [0,2,4]]
    data = DataVector(x=k,proj=proj,edges=[{'x':kedges}]*len(proj))
    data.view(xlim=(0.01,0.2))
    models = ModelCollection([model])
    cov = GaussianCovarianceMatrix(data,model_base=models.bases(),volume=(1e3)**3)
    cov.compute(models)
    assert cov.cov.shape == (data.get_x(concatenate=True).size,)*2
    cov.view(xlim=(0.01,0.1))
    filename = os.path.join(plot_dir,'covariance_pk.png')
    cov.plot(filename=filename,style='corr',data_styles='power')
    filename = os.path.join(plot_dir,'mean.png')
    cov.x[0].plot(filename=filename,style='power')

    data = MockDataVector(cov)
    filename = os.path.join(plot_dir,'real.png')
    data.plot(filename=filename,style='power')


if __name__ == '__main__':

    setup_logging()
    test_pk_multipole_covariance()
    test_pk_muwedge_covariance()
    #test_xi_covariance()
    #test_pk_xi_covariance()
    test_pk_covariance()
    test_pkell_covariance()
    test_view_covariance()
