import numpy as np
from scipy.interpolate import RegularGridInterpolator

from cosmopipe.lib.data_vector import DataVector


class WindowFunction(DataVector):

    def __call__(self, proj, *args, grid=True):
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
