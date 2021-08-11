"""Implementation of window functions."""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from cosmopipe.lib.data_vector import DataVector


class WindowFunction(DataVector):

    """Class representing a window function."""

    def __call__(self, proj, *args, grid=True):
        """
        Return projection ``proj`` of window function interpolated at input data points.

        Parameters
        ----------
        proj : ProjectionName
            Projection name.

        args : list
            List of coordinates where to interpolate window function.

        grid : bool
            Whether input coordinates should be interpreted as a grid,
            in which case the output will be an array of shape ``(x1.size, x2.size, ...)`` with ``x1``, ``x2`` the input coordinates.

        Returns
        -------
        toret : array
            Interpolate values of window function.
        """
        window = self.get(proj,permissive=True)
        if len(window) > 1:
            raise ValueError('Found several window functions for projection {}: {}'.format(proj,[w.proj for w in window]))
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
