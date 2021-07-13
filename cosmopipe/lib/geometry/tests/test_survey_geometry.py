import numpy as np

from cosmopipe.lib.geometry import PowerOddWideAngle
from cosmopipe.lib.geometry import PowerWindowMatrix


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


def test_powerodd_wideangle():
    ells = [0,2,4]
    kmin, kmax = 0.1,0.2
    nk = 3
    dk = (kmax - kmin)/nk
    k = np.array([i*dk + dk/2. for i in range(nk)])
    d = 1.
    wa = PowerOddWideAngle(ells,k=k,d=d,ellsout=range(5))

    from wide_angle_tools import get_end_point_LOS_M
    ref = get_end_point_LOS_M(d, Nkth=nk, kmin=kmin, kmax=kmax)

    test = np.empty_like(ref)
    for ikout in range(wa.matrix.shape[0]):
        for illout in range(wa.matrix.shape[1]):
            for ik in range(wa.matrix.shape[2]):
                for ill in range(wa.matrix.shape[3]):
                    test[ikout+illout*len(k),ik+ill*len(k)] = wa.matrix[ikout,illout,ik,ill]
                    if illout % 2 == 0: # in my code, even poles are zeros
                        test[ikout+illout*len(k),ik+ill*len(k)] = ref[ikout+illout*len(k),ik+ill*len(k)]

    #print('REF')
    #print(ref[nk:2*nk,nk:2*nk])
    #print('TEST')
    #print(test[nk:2*nk,nk:2*nk])
    assert np.allclose(test,ref)


def test_window_matrix():
    ells = (0,1,2,3,4)
    ellsout = ells
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
            if (ell % 2 == 1) and (n == 0): dwindow[n][ell][...] = 0.
            if (ell % 2 == 0) and (n == 1): dwindow[n][ell][...] = 0.
            #if (ell % 2 == 1): dwindow[n][ell][...] = 0.
            #if n >= 1: dwindow[n][ell][...] = 0.

    from scipy.interpolate import interp1d

    def window(s, ell=0):
        dwin = dwindow[ell % 2]
        #dwin = dwindow[0]
        if ell <= 4: dwin = dwin[ell]
        else: dwin = 0.*swin
        return interp1d(swin,dwin,kind='linear',fill_value=((1. if ell == 0 else 0.),0.),bounds_error=False)(s)

    wm = PowerWindowMatrix(ells,ellsout=ellsout,window=window,kout=kout,krange=None,srange=srange,ns=ns)
    k = wm.k
    mask = (k > 0.001) & (k < 1.)
    test = wm.matrix
    from create_Wll import create_W
    ref = create_W(kout,swin,dwindow)

    from matplotlib import pyplot as plt

    fig,lax = plt.subplots(len(ellsout),len(ells),figsize=(10,8))

    for illout,ellout in enumerate(ellsout):
        for ill,ell in enumerate(ells):
            #diff = test[:,illout,:,ill][0] - ref[(0,ellout,ell)][0]
            print(ellout,ell)
            test_, ref_ = test[0,illout,:,ill], ref[(0,ellout,ell)][0] + ref[(1,ellout,ell)][0]
            lax[illout][ill].plot(k[mask],test_[mask],label='test ({:d},{:d})'.format(ellout,ell))
            lax[illout][ill].plot(k[mask],ref_[mask],label='ref')
            lax[illout][ill].legend()
            #print(np.max(test_-ref_))
            #assert np.allclose(test_,ref_,rtol=1e-1,atol=1e-3)

    plt.show()





if __name__ == '__main__':

    #test_deriv()
    #test_powerodd_wideangle()
    test_window_matrix()
