import os

import numpy as np
from cosmoprimo import Cosmology

from cosmopipe.lib.theory import ProjectionBasis, LinearModel, ModelEvaluation, ModelCollection
from cosmopipe.lib.data_vector import ProjectionName, DataVector
from cosmopipe.lib.data_vector.plotting import PowerSpectrumPlotStyle
from cosmopipe.lib.survey_selection import PowerOddWideAngle, PowerWindowMatrix, ModelProjection, ModelProjectionCollection, WindowFunction
from cosmopipe.lib.survey_selection.binning import BaseBinning, InterpBinning
from cosmopipe.lib import setup_logging


base_dir = os.path.dirname(os.path.realpath(__file__))
plot_dir = os.path.join(base_dir,'_plots')
window_fn = os.path.join(plot_dir,'window_function.txt')


def test_deriv():
    n = 5
    m = np.zeros((n,n),dtype='f8')
    m += np.diag(np.ones(n-1),k=1) - np.diag(np.ones(n-1),k=-1)
    m[0,0] = -0.5
    m[0,1] = 0.5
    m[-1,-1] = 0.5
    m[-1,-2] = -0.5

    ref = np.zeros((n,n),dtype='f8')
    for index1 in range(n):
        index2 = index1

        if index2 > 0 and index2 < n-1:
            pre_factor = 1.
        else:
            pre_factor = 0.5

        if index2 > 0:
            ref[index1, index2-1] = -pre_factor
        else:
            ref[index1, index2] += -pre_factor

        if index2 < n-1:
            ref[index1, index2+1] = pre_factor
        else:
            ref[index1, index2] += pre_factor

    assert np.allclose(m,ref)


def test_power_odd_wideangle():
    ells = [0,2,4]
    kmin, kmax = 0.1,0.2
    nk = 3
    dk = (kmax - kmin)/nk
    k = np.array([i*dk + dk/2. for i in range(nk)])
    d = 1.
    wa = PowerOddWideAngle(d=1.,wa_orders=1,los='firstpoint')
    projsin = [ProjectionName(proj=ell,wa_order=0) for ell in ells]
    projsout = [ProjectionName(proj=ell,wa_order=ell%2) for ell in range(ells[-1]+1)]
    wa.setup(k,projsin,projsout=projsout)

    from wide_angle_tools import get_end_point_LOS_M
    ref = get_end_point_LOS_M(d, Nkth=nk, kmin=kmin, kmax=kmax)

    test = wa.matrix
    """
    test = np.empty_like(ref)
    for ikout in range(wa.matrix.shape[0]):
        for illout in range(wa.matrix.shape[1]):
            for ik in range(wa.matrix.shape[2]):
                for ill in range(wa.matrix.shape[3]):
                    test[ikout+illout*len(k),ik+ill*len(k)] = wa.matrix[ikout,illout,ik,ill]
                    if illout % 2 == 0: # in my code, even poles are zeros
                        test[ikout+illout*len(k),ik+ill*len(k)] = ref[ikout+illout*len(k),ik+ill*len(k)]
    """
    assert np.allclose(test,ref)


def test_window_matrix():
    ellsin = (0,1,2,3,4)
    wa_orders = [0,1]
    ellsout = ellsin
    ns = 1024*16
    #s = np.logspace(np.log10(srange[0]),np.log10(srange[1]),ns)
    kout = np.linspace(0.1,0.4,2)

    swin = np.linspace(1e-4,1e3,1000)
    srange = (swin[0],swin[-1])
    bwin = np.exp(-(swin/10.)**2)
    dwindow = {}
    for n in range(3):
        dwindow[n] = {}
        for ell in range(5):
            dwindow[n][ell] = bwin.copy()
            if ell > 0: dwindow[n][ell] *= np.random.uniform()
            #if (ell % 2 == 1) and (n == 0): dwindow[n][ell][...] = 0.
            #if (ell % 2 == 0) and (n == 1): dwindow[n][ell][...] = 0.
            #if (ell % 2 == 1): dwindow[n][ell][...] = 0.
            if n > 1: dwindow[n][ell][...] = 0.

    from scipy.interpolate import interp1d

    projsin = [ProjectionName(proj=ell,wa_order=wa_order) for ell in ellsin for wa_order in wa_orders]
    projsout = [ProjectionName(proj=ell,wa_order=wa_order) for ell in ellsout for wa_order in wa_orders]

    def window(proj, s):
        dwin = dwindow[proj.wa_order]
        ell = proj.proj
        if ell <= 4: dwin = dwin[ell]
        else: dwin = 0.*swin
        return interp1d(swin,dwin,kind='linear',fill_value=((1. if ell == 0 else 0.),0.),bounds_error=False)(s)

    wm = PowerWindowMatrix(window=window,krange=None,srange=srange,ns=ns,sum_wa=False)
    wm.setup([kout]*len(projsout),projsin,projsout=projsout)
    kin = wm.kin
    mask = (kin > 0.001) & (kin < 1.)
    test = wm.matrix

    from create_Wll import create_W
    ref = create_W(kout,swin,dwindow)

    from matplotlib import pyplot as plt
    from cosmopipe.lib.survey_selection.window_convolution import weights_trapz

    for wa_order in wa_orders:
        fig,lax = plt.subplots(len(ellsout),len(ellsin),figsize=(10,8))
        for illout,ellout in enumerate(ellsout):
            for illin,ellin in enumerate(ellsin):
                iprojout = projsout.index(ProjectionName(proj=ellout,wa_order=wa_order))
                iprojin = projsout.index(ProjectionName(proj=ellin,wa_order=wa_order))
                test_ = test[iprojout*len(kout),iprojin*len(kin):(iprojin+1)*len(kin)] / (weights_trapz(kin**3) / 3.)

                if wa_order % 2 == 0 and ellin % 2 == 1:
                    test_ = 0.*test_ # convention of create_Wll (no theory odd multipoles at 0th order)
                if wa_order % 2 == 1 and ellin % 2 == 0:
                    test_ = 0.*test_ # convention of create_Wll (no theory even multipoles at 1th order)
                if ellin % 2 == 1:
                    test_ *= -1 # convention for input odd power spectra (we provide the imaginary part, wide_angle_tools.py provides - the inmaginary part)
                if ellout % 2 == 1:
                    test_ *= -1 # same as above

                ref_ = ref[(wa_order,ellout,ellin)][0]
                lax[illout][illin].plot(kin[mask],test_[mask],label='test ({:d},{:d})'.format(ellout,ellin))
                lax[illout][illin].plot(kin[mask],ref_[mask],label='ref')
                lax[illout][illin].legend()
            #print(np.max(test_-ref_))
            #assert np.allclose(test_,ref_,rtol=1e-1,atol=1e-3)

    plt.show()


def save_window_function():

    swin = np.linspace(1e-4,1e3,1000)
    srange = (swin[0],swin[-1])
    bwin = np.exp(-(swin/100.)**2)
    y,projs = [],[]
    for n in range(2):
        for ell in range(9):
            y_ = bwin.copy()
            if ell > 0: y_ *= np.random.uniform()/10.
            y.append(y_)
            projs.append(ProjectionName(space=ProjectionName.CORRELATION,mode=ProjectionName.MULTIPOLE,proj=ell,wa_order=n))
    window = WindowFunction(x=[swin]*len(y),y=y,proj=projs)
    window.save_auto(window_fn)



def get_window_matrix():
    rebin_k = 2
    ns = rebin_k*1024
    srange = (1e-4,1e3)
    krange = (1e-3,1e2)
    """
    swin = np.linspace(1e-4,1e3,1000)
    srange = (swin[0],swin[-1])
    bwin = np.exp(-(swin/100.)**2)
    dwindow = {}
    for n in range(3):
        dwindow[n] = {}
        for ell in range(5):
            dwindow[n][ell] = bwin.copy()
            if ell > 0: dwindow[n][ell] *= np.random.uniform()/10.
            if n > 1: dwindow[n][ell][...] = 0.
            #if ell > 0: dwindow[n][ell][...] = 0.

    from scipy.interpolate import interp1d

    def window(proj, s):
        dwin = dwindow[proj.wa_order]
        ell = proj.proj
        if ell <= 4: dwin = dwin[ell]
        else: dwin = 0.*swin
        return interp1d(swin,dwin,kind='linear',fill_value=((1. if ell == 0 else 0.),0.),bounds_error=False)(s)
    """
    window = WindowFunction.load_auto(window_fn)
    return PowerWindowMatrix(window=window,krange=krange,srange=srange,ns=ns,rebin_k=rebin_k)


def test_window_convolution():

    pk = Cosmology().get_fourier('eisenstein_hu').pk_interpolator().to_1d()
    model = LinearModel(pklin=pk)
    k = np.linspace(0.02,0.2,40)
    projs = [ProjectionName(mode=ProjectionName.MULTIPOLE,proj=ell,wa_order=0) for ell in [0,2,4]]
    data = DataVector(x=k,proj=projs)

    wm = get_window_matrix()
    wm.setup(k,projsin=data.projs,projsout=data.projs)

    datanow = data.deepcopy()
    datanow.set_y(ModelEvaluation(data=data,model_bases=model.basis)(model,remove_shotnoise=True))

    x = DataVector(x=wm.xin,proj=wm.projsin)
    y = wm.compute(ModelEvaluation(data=x,model_bases=model.basis)(model,remove_shotnoise=True))
    dataw = data.deepcopy()
    dataw.set_y(y)

    style = PowerSpectrumPlotStyle(linestyles=['-','--'])
    filename = os.path.join(plot_dir,'window_convolution.png')
    style.plot([datanow,dataw],filename=filename)


def test_model_projection():

    pk = Cosmology().get_fourier('eisenstein_hu').pk_interpolator().to_1d()
    model = LinearModel(pklin=pk)
    k = np.linspace(0.02,0.2,20)
    data = DataVector(x=k,proj=['ell_0','ell_1','ell_2','ell_4'])
    list_data_vector = []

    model_projection = ModelProjection(data,model_bases=model.basis)
    model_projection.setup()
    #list_data_vector.append(model_projection.to_data_vector(model))
    # add BaseBinning
    model_projection.append(BaseBinning())
    model_projection.setup()
    list_data_vector.append(model_projection.to_data_vector(model))

    # add window function
    wm = get_window_matrix()
    model_projection.insert(0,wm)
    model_projection.setup()
    list_data_vector.append(model_projection.to_data_vector(model))
    # add wide-angle effects
    #wm = get_window_matrix()
    #model_projection.insert(0,wm)
    wa = PowerOddWideAngle(d=10.,wa_orders=1,los='firstpoint')
    model_projection.insert(0,wa)
    model_projection.setup()
    list_data_vector.append(model_projection.to_data_vector(model))
    """
    model_projection = ModelProjection(data,model_bases=model.basis)
    wa = PowerOddWideAngle(d=10.,wa_orders=1,los='firstpoint')
    model_projection.insert(0,wa)
    xin = model.basis.x
    model_projection.append(InterpBinning(xin=xin))
    model_projection.setup()
    list_data_vector.append(model_projection.to_data_vector(model))
    """
    style = PowerSpectrumPlotStyle(linestyles=['-','--',':'])
    filename = os.path.join(plot_dir,'projection.png')
    style.plot(list_data_vector,filename=filename)


def test_pkell_projection():

    pk = Cosmology().get_fourier('eisenstein_hu').pk_interpolator().to_1d()
    model = LinearModel(pklin=pk)
    data = DataVector(x=model.basis.x,proj=['ell_0','ell_2','ell_4'])
    projection = ModelProjection(data,model_bases=model.basis)
    projection.setup()
    pkell = projection(model,concatenate=False)
    from scipy import interpolate
    interp = interpolate.interp1d(model.basis.x,np.array(pkell).T,axis=0,kind='cubic',bounds_error=True,assume_sorted=True)
    interp.basis = ProjectionBasis({'x':model.basis.x,'space':'power','mode':'multipole','projs':[0,2,4],'wa_order':0})
    model = ModelCollection([interp])

    k = np.linspace(0.02,0.2,20)
    data = DataVector(x=k,proj=['ell_0','ell_1','ell_2','ell_4'])
    list_data_vector = []

    model_projection = ModelProjection(data,model_bases=model.bases())
    model_projection.setup()
    #list_data_vector.append(model_projection.to_data_vector(model))
    # add BaseBinning
    model_projection.append(BaseBinning())
    model_projection.setup()
    list_data_vector.append(model_projection.to_data_vector(model))

    # add window function
    wm = get_window_matrix()
    model_projection.insert(0,wm)
    model_projection.setup()
    list_data_vector.append(model_projection.to_data_vector(model))

    # add wide-angle effects
    #wm = get_window_matrix()
    #model_projection.insert(0,wm)
    wa = PowerOddWideAngle(d=10.,wa_orders=1,los='firstpoint')
    model_projection.insert(0,wa)
    model_projection.setup()
    list_data_vector.append(model_projection.to_data_vector(model))

    style = PowerSpectrumPlotStyle(linestyles=['-','--',':'])
    filename = os.path.join(plot_dir,'projection_pkell.png')
    style.plot(list_data_vector,filename=filename)



def test_model_projection_collection():

    pk = Cosmology().get_fourier('eisenstein_hu').pk_interpolator().to_1d()
    model = LinearModel(pklin=pk)
    k = np.linspace(0.02,0.2,20)
    data = DataVector(x=k,proj=['ell_0','ell_2','ell_4'])
    mc = ModelProjectionCollection(data,model_bases=model.basis)
    mc.setup()
    data_vector = mc.to_data_vector(model)
    filename = os.path.join(plot_dir,'projection_collection.png')
    data_vector.plot(filename=filename,style='power')


def test_copy():

    pk = Cosmology().get_fourier('eisenstein_hu').pk_interpolator().to_1d()
    model = LinearModel(pklin=pk)
    k = np.linspace(0.02,0.2,10)
    data = DataVector(x=k,proj=['ell_0','ell_1','ell_2','ell_4'])
    data_fine = DataVector(x=np.linspace(0.02,0.2,60),proj=['ell_0','ell_1','ell_2','ell_4'])

    model_projection = ModelProjection(data,model_bases=model.basis)
    # add window function
    wm = get_window_matrix()
    model_projection.insert(0,wm)
    model_projection.setup()
    model_projection_fine = model_projection.copy()
    model_projection_fine.setup(data_fine)

    list_data_vector = []
    list_data_vector.append(model_projection_fine.to_data_vector(model))
    list_data_vector.append(model_projection.to_data_vector(model))
    style = PowerSpectrumPlotStyle(linestyles=['-','--'])
    filename = os.path.join(plot_dir,'projection_model.png')
    style.plot(list_data_vector,filename=filename)

    # now with model collection
    mc = ModelProjectionCollection(data,model_bases=model.basis)
    mc_fine = mc.copy()
    mc_fine.setup(data_fine)
    mc.setup()

    list_data_vector = []
    list_data_vector.append(mc_fine.to_data_vector(model))
    list_data_vector.append(mc.to_data_vector(model))
    style = PowerSpectrumPlotStyle(linestyles=['-','--'])
    filename = os.path.join(plot_dir,'projection_collection_model.png')
    style.plot(list_data_vector,filename=filename)


if __name__ == '__main__':

    setup_logging()
    test_deriv()
    #test_power_odd_wideangle()
    #test_window_matrix()
    save_window_function()
    test_window_convolution()
    test_model_projection()
    test_pkell_projection()
    test_model_projection_collection()
    test_copy()
