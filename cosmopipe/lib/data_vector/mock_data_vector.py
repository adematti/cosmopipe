"""Definition of :class:`MockDataVector` to draw a data vector from a :class:`CovarianceMatrix`."""

import logging
import numpy as np

from cosmopipe.lib import mpi

from .data_vector import DataVector


class MockDataVector(DataVector):

    """Class to generate mock data vector, from an input covariance matrix."""

    def __init__(self, covariance, rng=None, seed=None, y=None, mean=False):
        """
        Initialize :class:`MockDataVector`.

        Parameters
        ----------
        covariance : CovarianceMatrix
            Input covariance matrix to draw samples from.

        rng : np.random.RandomState, default=None
            Random state, used if ``mean`` is ``False``.

        seed : int, default=None
            Random seed. Used if ``rng`` is ``None``.

        y : array, default=None
            If not ``None``, use as mean instead of ``covariance`` mean.

        mean : bool, default=False
            If ``True``, do not add noise to the mean.
        """
        if rng is None:
            seed = mpi.bcast_seed(seed=seed,mpicomm=self.mpicomm,size=None)
            rng = np.random.RandomState(seed=seed)

        data = covariance.x[0].deepcopy()
        if y is None:
            y = data.get_y()
        if not mean:
            y = rng.multivariate_normal(y,covariance.copy().noview().get_cov())
        data.set_y(y,concatenated=True)
        self.__dict__.update(data.__dict__)
        self.attrs.update(**covariance.attrs)
