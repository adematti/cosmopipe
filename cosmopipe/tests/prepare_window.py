import os

import numpy as np

from cosmopipe.lib import setup_logging
from cosmopipe.lib.data_vector import ProjectionName
from cosmopipe.lib.survey_selection import WindowFunction


def save_window_function(window_fn):

    swin = np.linspace(1e-4,1e3,1000)
    srange = (swin[0],swin[-1])
    bwin = np.exp(-(swin/100.)**2)
    y,projs = [],[]
    for n in range(2):
        for ell in range(9):
            y_ = bwin.copy()
            if ell > 0: y_ *= np.random.uniform()/10.
            y.append(y_)
            projs.append(ProjectionName(space=ProjectionName.CORRELATION,mode=ProjectionName.MULTIPOLE,proj=ell,wa_order=n))
    window = WindowFunction(x=[swin]*len(y),y=y,proj=projs)
    window.save_auto(window_fn)


if __name__ == '__main__':

    setup_logging()
    base_dir = '_data'
    window_fn = os.path.join(base_dir,'window_function.txt')
    save_window_function(window_fn)
