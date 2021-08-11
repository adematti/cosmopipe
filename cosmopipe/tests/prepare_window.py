import os

import numpy as np

from cosmopipe.lib import setup_logging
from cosmopipe.lib.data_vector import ProjectionName
from cosmopipe.lib.survey_selection import WindowFunction


def save_synthetic_window_function(window_fn):

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


def save_power_window_function(data_fn, randoms_fn, window_fn):

    from scipy import special
    from cosmopipe.estimators.utils import prepare_survey_catalogs
    from cosmopipe.lib.catalog import Catalog
    from cosmopipe.lib import mpi

    data = Catalog.load_fits(data_fn,mpistate='scattered')
    randoms = Catalog.load_fits(randoms_fn,mpistate='scattered')
    data, randoms = prepare_survey_catalogs(data,randoms,cosmo=None,position='Position',nbar='NZ')
    data['weight'] = data['weight_fkp']*data['weight_comp']
    randoms['weight'] = randoms['weight_fkp']*randoms['weight_comp']
    alpha = data.sum('weight')/randoms.sum('weight')
    # this is the normalization factor A used in the power spectrum estimation, either computed from data or randoms
    norm = alpha * mpi.sum_array(randoms['weight']*randoms['nbar']*randoms['weight_fkp'],mpicomm=randoms.mpicomm)
    #norm = mpi.sum_array(data['weight']*data['nbar']*data['weight_fkp'],mpicomm=randoms.mpicomm)
    #exit()

    from nbodykit.lab import FKPCatalog, ConvolvedFFTPower
    # There is some dependence in both the mesh resolution and box size
    # Higher mesh resolution increases window function (more power on small scales)
    # Larger box size yields better convergence on large scales
    # There must be a correct choice, I should look again at Pat's notes
    BoxSize = 10000.
    Nmesh = 256
    ells = (0,2,4)
    # We can try downsampling or rescaling random weights, result is the same
    #randoms = randoms.gslice(0,-1,10)
    #randoms['weight_comp'] *= 10
    randoms['weight'] = randoms['weight_fkp']*randoms['weight_comp']
    # The 3 next lines simply compute the power spectrum of the random catalog
    # The only difference w.r.t. the power spectrum estimation of data - randoms, is that the normalization A is 1.
    fkp = FKPCatalog(randoms.to_nbodykit(),None,nbar='nbar')
    mesh = fkp.to_mesh(position='position',fkp_weight='weight_fkp',comp_weight='weight_comp',nbar='nbar',BoxSize=BoxSize,Nmesh=Nmesh,resampler='tsc',interlaced=True,compensated=True)
    power = ConvolvedFFTPower(mesh,poles=ells,kmin=0.,dk=None)
    kfun = 2.*np.pi/mesh.attrs['BoxSize']
    poles = power.poles
    mask = poles['k'] > 0.
    k = poles['k'][mask]
    volume = np.prod(kfun)*poles['modes'][mask]

    normwin = randoms.sum('weight')**2/data.sum('weight')**2*norm # this is 1 / alpha^2 * A
    shotnoise = mpi.sum_array(randoms['weight']**2,mpicomm=randoms.mpicomm)/normwin

    def Wk(ell):
        if ell == 0: return poles['power_{:d}'.format(ell)].real[mask]/normwin - shotnoise
        return poles['power_{:d}'.format(ell)].real[mask]/normwin

    swin = np.linspace(0,6e3,1000)
    kk,ss = np.meshgrid(k,swin,sparse=False,indexing='ij')
    ks = kk*ss

    y,projs = [],[]
    # let us compute the window function in configuration space
    for ell in ells:
        integrand = Wk(ell)[:,None]*1./(2.*np.pi)**3 * special.spherical_jn(ell,ks)
        prefactor = (1j) ** ell # this is definition of the Hankel transform
        if ell % 2 == 1: prefactor *= -1j # we provide the imaginary part of odd power spectra, so let's multiply by (-i)^ell
        prefactor = np.real(prefactor)
        y.append(prefactor * np.sum(volume[:,None]*integrand,axis=0)) # this is the multipole ell in configuration space
        projs.append(ProjectionName(space=ProjectionName.CORRELATION,mode=ProjectionName.MULTIPOLE,proj=ell,wa_order=0))
    window = WindowFunction(x=[swin]*len(y),y=y,proj=projs,attrs={'norm':norm})
    #window.save_auto(window_fn)

    from matplotlib import pyplot as plt
    for proj in window.get_projs():
        plt.plot(window.get_x(proj=proj),window.get_y(proj=proj),label=proj.get_projlabel())
    plt.axhline(1.,0.,1.,linestyle='--',color='k')
    plt.xscale('log')
    plt.legend()
    plt.show()


if __name__ == '__main__':

    setup_logging()
    base_dir = '_catalog'
    data_fn = os.path.join(base_dir,'lognormal_data.fits')
    randoms_fn = os.path.join(base_dir,'lognormal_randoms.fits')
    window_fn = os.path.join(base_dir,'window_function.txt')
    #save_synthetic_window_function(window_fn)
    save_power_window_function(data_fn,randoms_fn,window_fn)
