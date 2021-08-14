"""Definitions of specialized catalog classes, e.g. random catalog in box."""

import logging

import numpy as np
from cosmopipe.lib import utils, mpi

from .base import BaseCatalog


class Catalog(BaseCatalog):

    """Class that represents a standard catalog."""
    logger = logging.getLogger('Catalog')


class RandomBoxCatalog(Catalog):

    """Class that builds a random catalog with box geometry."""
    logger = logging.getLogger('RandomBoxCatalog')

    @mpi.MPIInit
    def __init__(self, BoxSize=1., BoxCenter=0., size=None, nbar=None, rng=None, seed=None, attrs=None):
        """
        Initialize :class:`RandomBoxCatalog`.

        Parameters
        ----------
        BoxSize : float, array
            3D physical extent of catalog.

        BoxCenter : float, array
            3D center of catalog.

        size : int, default=None
            Catalog global size.
            If ``None``, size is a Poisson draw of ``nbar * BoxSize``.

        nbar : float, default=None
            Density. Must be provided is ``size`` is ``None``.

        rng : mpi.MPIRandomState, numpy.random.RandomState
            Random state. Only :class:`mpi.MPIRandomState` garantees invariance with number of ranks.
            If ``None``, a new :class:`mpi.MPIRandomState` is created from ``seed``.

        seed : int, default=None
            Random seed to use when initializing new :class:`mpi.MPIRandomState`.

        attrs : dict
            Other attributes.
        """
        boxsize = np.empty(3,dtype='f8')
        boxsize[:] = BoxSize
        boxcenter = np.empty(3,dtype='f8')
        boxcenter[:] = BoxCenter
        self.rng = rng
        if size is None:
            if rng is None: rng = np.random.RandomState(seed=seed)
            size = rng.poisson(nbar*np.prod(boxsize))
            size = mpi.local_size(size,mpicomm=self.mpicomm)
        if self.rng is None:
            self.rng = mpi.MPIRandomState(mpicomm=self.mpicomm,seed=seed,size=size)
        position = np.array([rng.uniform(-boxsize[i]/2.+boxcenter[i],boxsize[i]/2.+boxcenter[i],size=size) for i in range(3)]).T
        attrs = attrs or {}
        attrs['BoxSize'] = boxsize
        attrs['BoxCenter'] = boxcenter
        mpistate = self.mpistate
        super(RandomBoxCatalog,self).__init__(data={'Position':position},mpicomm=self.mpicomm,mpistate='scattered',mpiroot=self.mpiroot,attrs=attrs)
        self.mpi_to_state(mpistate=mpistate)
