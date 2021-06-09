import logging
import numpy as np

from cosmopipe.lib import mpi

from .data_vector import DataVector


class MockDataVector(DataVector):

    logger = logging.getLogger('MockDataVector')

    def __init__(self, covariance, seed=None, rng=None, mean=False):

        if rng is None:
            seed = mpi.bcast_seed(seed=seed,mpicomm=self.mpicomm,size=None)
            rng = np.random.RandomState(seed=seed)

        y = covariance.x[0].y
        if not mean:
            y = rng.multivariate_normal(y,covariance.cov)
        super(MockDataVector,self).__init__(x=covariance.x[0].x,y=y,proj=covariance.x[0].proj,**covariance.attrs)
