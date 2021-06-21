import os

import numpy as np

from cosmopipe.lib.data import DataVector, CovarianceMatrix, MockCovarianceMatrix, MockDataVector
from cosmopipe.lib.data.plotting import PowerSpectrumPlotStyle
from cosmopipe.lib.data.projection import ProjectionName
from cosmopipe.lib import utils, setup_logging


base_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(base_dir,'_data')
data_fn = os.path.join(data_dir,'_data_{:d}.txt')
covariance_fn = os.path.join(data_dir,'covariance.txt')


def make_data_covariance(data_fn=data_fn,covariance_fn=covariance_fn,mapping_proj=None,ndata=30,seed=42):
    utils.mkdir(os.path.dirname(data_fn))
    utils.mkdir(os.path.dirname(covariance_fn))
    x = np.linspace(0.01,0.2,5)
    rng = np.random.RandomState(seed=seed)
    list_data = []
    for i in range(ndata):
        y = [(rng.uniform(-100.,100.,size=x.size)+1000*i)/x for i in reversed(range(len(mapping_proj)))]
        data = DataVector(x=x,y=y,mapping_proj=mapping_proj)
        with open(data_fn.format(i),'w') as file:
            file.write('#Estimated shot noise: 3000.0\n')
            template = ' '.join(['{:.18e}']*(len(y)+1)) + '\n'
            y = np.array(y).T
            for ix,x_ in enumerate(x):
                file.write(template.format(x_,*y[ix]))
        list_data.append(data)
    if ndata <= 1:
        return list_data,None
    cov = MockCovarianceMatrix.from_data(list_data)
    with open(covariance_fn,'w') as file:
        file.write('#Nobs: {:d}\n'.format(ndata))
        template = '{:d} {:d} {:.18e}\n'
        for i in range(cov.shape[0]):
            for j in range(cov.shape[1]):
                file.write(template.format(i,j,cov._covariance[i,j]))
    return list_data,cov


def test_multipole_data_vector():

    mapping_proj = ['ell_0','ell_2','ell_4']

    list_data = make_data_covariance(ndata=1,mapping_proj=mapping_proj)[0]
    mapping_header = {'shotnoise':'.*?Estimated shot noise: (.*)'}
    data = DataVector.load_txt(data_fn.format(0),mapping_header=mapping_header,mapping_proj=mapping_proj)
    assert np.allclose(data.get_x(),list_data[0].get_x())
    assert np.allclose(data.get_x(proj='ell_0'),list_data[0].get_x(proj='ell_0'))
    assert np.all(data.view(proj='ell_0').get_y() == data.get_y(proj='ell_0'))
    assert data.attrs['shotnoise'] == 3000.
    filename = os.path.join(data_dir,'data.npy')
    data.save(filename)
    data2 = DataVector.load(filename)
    assert np.all(data2.get_x(proj='ell_2') == data.get_x(proj='ell_2'))
    filename = os.path.join(data_dir,'data.txt')
    data.save_txt(filename)
    data2 = DataVector.load_txt(filename)
    assert np.all(data2.get_x(proj='ell_4') == data.get_x(proj='ell_4'))
    filename = os.path.join(data_dir,'plot_data.png')
    data2.plot(filename=filename,style='pk')
    mapping_proj = ['ell_0']
    list_data = make_data_covariance(ndata=1,mapping_proj=mapping_proj)[0]
    mapping_header = {'shotnoise':'.*?Estimated shot noise: (.*)'}
    data = DataVector.load_txt(data_fn.format(0),mapping_header=mapping_header)
    assert np.allclose(data.get_x(),list_data[0].get_x())
    assert data.attrs['shotnoise'] == 3000.
    filename = os.path.join(data_dir,'data.npy')
    data.save(filename)
    data2 = DataVector.load(filename)
    assert np.all(data2.get_x() == data.get_x())
    filename = os.path.join(data_dir,'data.txt')
    data.save_txt(filename)
    data2 = DataVector.load_txt(filename, type='pk')
    assert np.all(data2.get_x() == data.get_x())
    filename = os.path.join(data_dir,'plot_data_0.png')
    data2.plot(filename=filename)


def test_multipole_covariance_matrix():

    mapping_proj = ['ell_0','ell_2','ell_4']
    list_data,cov_ref = make_data_covariance(ndata=60,mapping_proj=mapping_proj)
    cov = CovarianceMatrix.load_txt(covariance_fn)
    assert np.allclose(cov.get_cov(),cov_ref.get_cov())
    cov2 = CovarianceMatrix.load_txt(covariance_fn,data=list_data[0])
    assert np.allclose(cov2.get_cov(),cov_ref.get_cov())
    assert np.allclose(cov2.get_x()[0],cov_ref.get_x()[0])
    filename = os.path.join(data_dir,'covariance.npy')
    cov2.save(filename)
    cov2 = CovarianceMatrix.load(filename)
    assert np.all(cov2.get_cov() == cov.get_cov())
    filename = os.path.join(data_dir,'covariance.txt')
    cov2.save_txt(filename)
    cov2 = CovarianceMatrix.load_txt(filename)
    assert np.allclose(cov2.get_cov(),cov.get_cov())
    for kwargs in [{'block':False},{'block':True},{'block':True,'proj':['ell_0','ell_2','ell_4']}]:
        assert np.allclose(cov2.get_cov().dot(cov2.get_invcov(**kwargs)),np.eye(*cov2.shape))
    filename = os.path.join(data_dir,'plot_covariance.png')
    cov2.plot(filename=filename,data_styles='pk')

    mapping_proj = ['ell_0']
    list_data,cov_ref = make_data_covariance(ndata=60,mapping_proj=mapping_proj)
    data = DataVector.load_txt(data_fn.format(0))
    cov = CovarianceMatrix.load_txt(covariance_fn,data=data,type='pk')
    assert np.allclose(cov.get_cov(),cov_ref.get_cov())
    filename = os.path.join(data_dir,'plot_covariance_0.png')
    cov.plot(filename=filename)


def test_mock_data_vector():
    cov = make_data_covariance(ndata=60,mapping_proj=['ell_0','ell_2'])[1]
    data = MockDataVector(cov,mean=True)
    assert np.allclose(data.x,cov.x[0].x)
    assert np.allclose(data.y,cov.x[0].y)
    data = MockDataVector(cov,seed=42)
    assert not np.allclose(data.y,cov.x[0].y)


def test_plotting():
    mapping_proj = ['ell_0','ell_2','ell_4']
    list_data,cov_ref = make_data_covariance(ndata=10,mapping_proj=mapping_proj)
    style = PowerSpectrumPlotStyle()
    #style.plot(list_data,filename=os.path.join(data_dir,'plot_list_data.png'))
    style.plot(list_data,covariance=cov_ref,filename=os.path.join(data_dir,'plot_list_data.png'))


def test_muwedge_data_vector():
    mapping_proj = [('muwedge',(0.,1./3.)),('muwedge',(1./3.,2./3.))]
    list_data = make_data_covariance(ndata=1,mapping_proj=mapping_proj)[0]
    mapping_header = {'shotnoise':'.*?Estimated shot noise: (.*)'}
    data = DataVector.load_txt(data_fn.format(0),mapping_header=mapping_header,mapping_proj=mapping_proj)
    assert np.allclose(data.get_x(),list_data[0].get_x())
    muwedge0 = mapping_proj[0]
    assert np.allclose(data.get_x(proj=muwedge0),list_data[0].get_x(proj=muwedge0))
    assert np.all(data.view(proj=muwedge0).get_y() == data.get_y(proj=muwedge0))
    assert data.attrs['shotnoise'] == 3000.
    filename = os.path.join(data_dir,'data.npy')
    data.save(filename)
    data2 = DataVector.load(filename)
    assert np.all(data2.get_x(proj=muwedge0) == data.get_x(proj=muwedge0))
    filename = os.path.join(data_dir,'data.txt')
    data.save_txt(filename)
    data2 = DataVector.load_txt(filename)
    assert np.all(data2.get_x(proj=muwedge0) == data.get_x(proj=muwedge0))
    filename = os.path.join(data_dir,'plot_data.png')
    data2.plot(filename=filename,style='pk')


def test_projection():
    ell0 = ProjectionName('ell_0')
    assert ell0.mode == 'multipole' and ell0.proj == 0
    mu0 =  ProjectionName('mu_0_0.5')
    assert mu0.mode == 'muwedge' and mu0.proj == (0.,0.5)
    mu0 = ProjectionName('mu_0_1/3')
    assert mu0.mode == 'muwedge' and mu0.proj == (0.,1./3)


if __name__ == '__main__':

    setup_logging()
    #test_multipole_data_vector()
    #test_multipole_covariance_matrix()
    #test_mock_data_vector()
    #test_plotting()
    #test_muwedge_data_vector()
    test_projection()
