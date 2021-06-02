import numpy as np
from scipy import interpolate


class RedshiftDensityInterpolator(ScatteredBaseClass):

    logger = logging.getLogger('RedshiftDensityInterpolator')

    @mpi.MPIInit
    def __init__(self, redshifts, weights=None, bins=None, fsky=1., radial_distance=None, interp_order=1):
        """
        Inspired from: https://github.com/bccp/nbodykit/blob/master/nbodykit/algorithms/zhist.py
        """
        def zrange(redshifts):
            if self.is_mpi_scattered():
                return mpi.min_array(redshifts,mpicomm=self.mpicomm),mpi.max_array(redshifts,mpicomm=self.mpicomm)
            else:
                zrange = None
                if self.is_mpi_root():
                    zrange = np.min(redshifts),np.max(redshifts)
                return self.mpicomm.bcast(zrange,root=self.mpiroot)

        if bins is None or bins == 'scott':
            # scott's rule
            if self.is_mpi_scatter():
                var = mpi.var_array(redshifts,aweights=weights,ddof=1,mpicomm=self.mpicomm)
                gsize = mpi.size_array(redshifts)
            else:
                var,gsize = None,None
                if self.is_mpi_root():
                    var = np.cov(redshifts,aweights=weights,ddof=1)
                    gsize = redshifts.size
                var,gsize = self.mpicomm.bcast((var,gsize),root=self.mpiroot)
            sigma = np.sqrt(var)
            dz = sigma * (24. * np.sqrt(np.pi) / gsize) ** (1. / 3)
            zrange = zrange(redshifts)
            nbins = np.ceil((maxval - minval) * 1. / dx)
            nbins = max(1, Nbins)
            edges = minval + dx * np.arange(nbins + 1)

        if np.ndim(bins) == 0:
            bins = np.linspace(*zrange(redshifts),num=bins+1,endpoint=True)

        counts = np.histogram(redshifts,weights=weights,bins=bins)
        if self.is_mpi_scattered():
            counts = mpicomm.allreduce(counts,op=mpi.MPI.SUM)

        if radial_distance is not None:
            dbins = radial_distance(bins)
        else:
            dbins = bins
        dvol = fsky*4./3.*np.pi*(dbins[1:]**3 - dbins[:-1]**3)
        self.z = (bins[:-1] + bins[1:])/2.
        self.density = counts/dvol
        self.spline = interpolate.UnivariateSpline(self.z,self.density,k=interp_order,s=0)

    def __call__(self, z):
        return self.spline(z)


def distance(position):
	return np.sqrt((position**2).sum(axis=-1))


def cartesian_to_sky(position, wrap=True, degree=True):
	"""Transform cartesian coordinates into distance, RA, Dec.

	Parameters
	----------
	position : array of shape (N,3)
		position in cartesian coordinates.
	wrap : bool, optional
		whether to wrap ra into [0,2*pi]
	degree : bool, optional
		whether RA, Dec are in degree (True) or radian (False).

	Returns
	-------
	dist : array
		distance.
	ra : array
		RA.
	dec : array
		Dec.
	"""
	dist = distance(position)
	ra = np.arctan2(position[:,1],position[:,0])
	if wrap: ra %= 2.*np.pi
	dec = np.arcsin(position[:,2]/dist)
    conversion = np.pi/180. if degree else 1.
	return dist, ra/conversion, dec/conversion


def sky_to_cartesian(dist, ra, dec, degree=True, dtype=None):
	"""Transform distance, RA, Dec into cartesian coordinates.

	Parameters
	----------
	dist : array
		distance.
	ra : array
		RA.
	dec : array
		Dec.
	degree : bool
		whether RA, Dec are in degree (True) or radian (False).
	dtype : dtype, optional
		return array dtype.

	Returns
	-------
	position : array
		position in cartesian coordinates; of shape (len(dist),3).
	"""
	conversion = 1.
	if degree: conversion = np.pi/180.
	position = [None]*3
	cos_dec = np.cos(dec*conversion)
	position[0] = cos_dec*np.cos(ra*conversion)
	position[1] = cos_dec*np.sin(ra*conversion)
	position[2] = np.sin(dec*conversion)
	return (dist*np.asarray(position,dtype=dtype)).T
