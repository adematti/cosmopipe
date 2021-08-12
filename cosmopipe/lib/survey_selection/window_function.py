"""Implementation of window functions."""

import numpy as np
from scipy import special
from scipy.interpolate import RegularGridInterpolator

from cosmopipe.lib.data_vector import BinnedProjection, DataVector, ProjectionName


def compute_real_window_1d(s, fourier_window):
    """
    Compute configuration-space window function by taking Hankel transforms
    of Fourier-space projections of input window function.

    Note
    ----
    If the number of modes (as :class:`BinnedProjection` "weights"), as well as the physical size of the box
    used to compute the window function (as ``BoxSize`` attribute in :attr:`BinnedProjection.attrs`) are provided,
    use these to compute the Fourier-volume element in the Hankel transform.
    Else, use :attr:`BinnedProjection.edges` if provided, else differences in x-coordinates (:math:`k`) to compute the Fourier-volume element.

    Parameters
    ----------
    s : array
        Separations :math:`s` where to compute Hankel transform.

    fourier_window : WindowFunction, DataVector
        Window function. Only Fourier-space projections for which (:attr:`ProjectionName.space` is :attr:`ProjectionName.POWER`)
        are Hankel-transformed.

    Returns
    -------
    window : WindowFunction
        Window function with only configuration-space projections (:attr:`ProjectionName.space` is :attr:`ProjectionName.CORRELATION`).
    """
    real_window = WindowFunction()
    logged = False
    for proj in fourier_window.projs.select(space=ProjectionName.POWER,mode=ProjectionName.MULTIPOLE):
        dataproj = fourier_window.get(proj)
        k = dataproj.get_x()
        wk = dataproj.get_y()
        mask = k > 0
        k, wk = k[mask], wk[mask]
        if dataproj.has_weights() and 'BoxSize' in dataproj.attrs:
            nmodes = dataproj.get_weights()
            nmodes = nmodes[mask]
            boxsize = np.empty(3,dtype='f8')
            boxsize[:] = dataproj.attrs['BoxSize']
            volume = (2.*np.pi)**3/np.prod(boxsize)*nmodes
        else:
            if not logged:
                dataproj.log_info('Missing modes or BoxSize to compute volume in descrete limit. Switching to continuous limit.',rank=0)
                logged = True
            if dataproj.has_edges():
                edges = dataproj.get_edges()
                volume = 4./3.*np.pi*(edges[1:]**3 - edges[:-1]**3)
            else:
                dk = np.diff(k)
                dk = np.append(dk,dk[-1])
                volume = 4./3.*k**2*dk
            volume = volume[mask]

        kk,ss = np.meshgrid(k,s,indexing='ij')
        ks = kk*ss
        integrand = wk[:,None]*1./(2.*np.pi)**3 * special.spherical_jn(proj.proj,ks)
        prefactor = (1j) ** proj.proj # this is definition of the Hankel transform
        if proj.proj % 2 == 1: prefactor *= -1j # we provide the imaginary part of odd power spectra, so let's multiply by (-i)^ell
        prefactor = np.real(prefactor)
        y = prefactor * np.sum(volume[:,None]*integrand,axis=0)
        proj = proj.copy(space=ProjectionName.CORRELATION)
        dataproj = BinnedProjection(data={'s':s,'corr':y},x='s',y='corr',edges={'s':np.append(s,s[-1])},proj=proj,attrs=dataproj.attrs.copy())
        real_window.set(dataproj)

    return real_window


class WindowFunction(DataVector):

    """Class representing a window function."""

    def __call__(self, proj, *args, grid=True, default_zero=False):
        """
        Return projection ``proj`` of window function interpolated at input data points.

        Parameters
        ----------
        proj : ProjectionName
            Projection name.

        args : list
            List of coordinates where to interpolate window function.

        grid : bool, default=True
            Whether input coordinates should be interpreted as a grid,
            in which case the output will be an array of shape ``(x1.size, x2.size, ...)`` with ``x1``, ``x2`` the input coordinates.

        default_zero : bool, default=False
            If a given projection is not provided in window function, set to 0.
            Else an :class:`IndexError` is raised.

        Returns
        -------
        toret : array
            Interpolated values of window function.
        """
        window = self.get(proj,permissive=True)
        if len(window) > 1:
            raise ValueError('Found several window functions for projection {}: {}'.format(proj,[w.proj for w in window]))
        if not window:
            if default_zero:
                self.log_info('No window provided for projection {}, defaulting to 0.'.format(proj),rank=0)
                if grid:
                    return np.zeros(tuple(a.size for a in args),dtype=args[0].dtype)
                return np.zeros_like(args[0])
            else:
                raise IndexError('No window provided for projection {}'.format(proj))
        window = window[0]
        points = window.get_x_average()
        values = window.get_y(flatten=False)
        interp = RegularGridInterpolator(points,values,method='linear',bounds_error=False,fill_value=0.) # we would want input window function to start at 0.
        if grid:
            shape = [a.size for a in args]
            xy = np.meshgrid(args,indexing='ij') # super memory-consuming, but just not the option in scipy, should be easy to recode
            xy = np.array([xy_.flatten() for xy_ in xy]).T
        else:
            xy = np.array([xy_.flatten() for xy_ in args]).T
        toret = interp(xy)
        if grid:
            toret = toret.reshape(shape)
        return toret
