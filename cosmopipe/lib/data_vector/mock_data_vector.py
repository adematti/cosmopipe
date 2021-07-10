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

        data = covariance.x[0].deepcopy()
        y = data.get_y()
        if not mean:
            y = rng.multivariate_normal(y,covariance.copy().noview().get_cov())
        data.set_y(y,concatenated=True)
        self.__dict__.update(data.__dict__)
        self.attrs.update(**covariance.attrs)
