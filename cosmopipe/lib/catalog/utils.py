"""Utilities related to data clustering catalogs."""

import logging

import numpy as np
from scipy import interpolate

from cosmopipe.lib.utils import ScatteredBaseClass
from cosmopipe.lib import mpi


def vector_projection(vector, direction):
    r"""
    Vector components of given vectors in a given direction.

    .. math::
       \mathbf{v}_\mathbf{d} &= (\mathbf{v} \cdot \hat{\mathbf{d}}) \hat{\mathbf{d}} \\
       \hat{\mathbf{d}} &= \frac{\mathbf{d}}{\|\mathbf{d}\|}

    Adapted from https://github.com/bccp/nbodykit/blob/master/nbodykit/transform.py

    Parameters
    ----------
    vector : array
        Array of vectors to be projected (along last dimension).

    direction : array
        Projection direction, 1D or 2D (if different direction for each input ``vector``) array.
        It will be normalized.

    Returns
    -------
    projection : array
        Vector components of the given vectors in the given direction.
        Same shape as input ``vector``.
    """
    direction = np.asarray(direction, dtype='f8')
    direction = direction / (direction ** 2).sum(axis=-1)[:, None] ** 0.5
    projection = (vector * direction).sum(axis=-1)
    projection = projection[:, None] * direction

    return projection


class DistanceToRedshift(object):

    """Class that holds a conversion distance -> redshift."""

    def __init__(self, distance, zmax=100., nz=2048, interp_order=3):
        """
        Initialize :class:`DistanceToRedshift`.
        Creates an array of redshift -> distance in log(redshift) and instantiates
        a spline interpolator distance -> redshift.

        Parameters
        ----------
        distance : callable
            Callable that provides distance as a function of redshift (array).

        zmax : float, default=100.
            Maximum redshift for redshift <-> distance mapping.

        nz : int, default=2048
            Number of points for redshift <-> distance mapping.

        interp_order : int, default=3
            Interpolation order, e.g. ``1`` for linear interpolation, ``3`` for cubic splines.
        """
        self.distance = distance
        self.zmax = zmax
        self.nz = nz
        zgrid = np.logspace(-8,np.log10(self.zmax),self.nz)
        self.zgrid = np.concatenate([[0.], zgrid])
        self.rgrid = self.distance(self.zgrid)
        self.interp = interpolate.UnivariateSpline(self.rgrid,self.zgrid,k=interp_order,s=0)

    def __call__(self, distance):
        """Return (interpolated) redshift at distance ``distance`` (scalar or array)."""
        return self.interp(distance)


class RedshiftDensityInterpolator(ScatteredBaseClass):
    """
    Class that computes and interpolates a redshift density histogram :math:`n(z)` from an array of redshift and optionally weights.
    Adapted from: https://github.com/bccp/nbodykit/blob/master/nbodykit/algorithms/zhist.py
    """

    @mpi.MPIInit
    def __init__(self, redshifts, weights=None, bins=None, fsky=1., radial_distance=None, interp_order=1):
        r"""
        Initialize :class:`RedshiftDensityInterpolator`.

        Parameters
        ----------
        redshifts : array
            Array of redshifts.

        weights : array, default=None
            Array of weights, same shape as ``redshifts``. Defaults to 1.

        bins : int, array, string, default=None
            If `bins` is an integer, it defines the number of equal-width
            bins in the given range. If `bins` is a sequence, it defines the bin
            edges, including the rightmost edge, allowing for non-uniform bin widths.
            If 'scott', Scott's rule is used to estimate the optimal bin width
            from the input data. Defaults to 'scott'.

        fsky : float, default=1
            The sky area fraction, which is used in the volume calculation when normalizing :math:`n(z)`.
            ``1`` corresponds to full-sky: :math:`4 \pi` or :math:`\simeq 41253\; \mathrm{deg}^{2}`.

        radial_distance : callable, default=None
            Radial distance to use when converting redshifts into comoving distance.
            If ``None``, ``redshifts`` and optionally ``bins`` are assumed to be in distance units.

        interp_order : int, default=1
            Interpolation order, e.g. ``1`` for linear interpolation, ``3`` for cubic splines.
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
            nbins = max(1, nbins)
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
        """Return density at redshift ``z`` (scalar or array)."""
        return self.spline(z)


def distance(position):
    """Return cartesian distance, taking coordinates along ``position`` last axis."""
    return np.sqrt((position**2).sum(axis=-1))


def cartesian_to_sky(position, wrap=True, degree=True):
    r"""
    Transform cartesian coordinates into distance, RA, Dec.

    Parameters
    ----------
    position : array of shape (N,3)
        Position in cartesian coordinates.

    wrap : bool, default=True
        Whether to wrap RA in :math:`[0, 2 \pi]`.

    degree : bool, default=True
        Whether RA, Dec are in degree (``True``) or radian (``False``).

    Returns
    -------
    dist : array
        Distance.

    ra : array
        Right Ascension.

    dec : array
        Declination.
    """
    dist = distance(position)
    ra = np.arctan2(position[:,1],position[:,0])
    if wrap: ra %= 2.*np.pi
    dec = np.arcsin(position[:,2]/dist)
    conversion = np.pi/180. if degree else 1.
    return dist, ra/conversion, dec/conversion


def sky_to_cartesian(dist, ra, dec, degree=True, dtype=None):
    """
    Transform distance, RA, Dec into cartesian coordinates.

    Parameters
    ----------
    dist : array
        Distance.

    ra : array
        Right Ascension.

    dec : array
        Declination.

    degree : default=True
        Whether RA, Dec are in degree (``True``) or radian (``False``).

    dtype : numpy.dtype, default=None
        :class:`numpy.dtype` for returned array.

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
