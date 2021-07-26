import logging

import numpy as np
from cosmopipe.lib import utils, mpi

from .base import BaseCatalog


class Catalog(BaseCatalog):

    logger = logging.getLogger('Catalog')


class RandomCatalog(Catalog):

    logger = logging.getLogger('RandomCatalog')

    @mpi.MPIInit
    def __init__(self, BoxSize=1., BoxCenter=0., size=None, nbar=None, rng=None, seed=None, attrs=None):
        boxsize = np.empty(3,dtype='f8')
        boxsize[:] = BoxSize
        boxcenter = np.empty(3,dtype='f8')
        boxcenter[:] = BoxCenter
        if rng is None:
            seed = mpi.bcast_seed(seed=seed,mpicomm=self.mpicomm)
            rng = np.random.RandomState(seed=seed)
        self.rng = rng
        if size is None:
            size = rng.poisson(nbar*np.prod(boxsize))
        size = mpi.local_size(size,mpicomm=self.mpicomm)
        position = np.array([rng.uniform(-boxsize[i]/2.+boxcenter[i],boxsize[i]/2.+boxcenter[i],size=size) for i in range(3)]).T
        attrs = attrs or {}
        attrs['BoxSize'] = boxsize
        attrs['BoxCenter'] = boxcenter
        mpistate = self.mpistate
        super(RandomCatalog,self).__init__(data={'Position':position},mpicomm=self.mpicomm,mpistate='scattered',mpiroot=self.mpiroot,attrs=attrs)
        self.mpi_to_state(mpistate=mpistate)
