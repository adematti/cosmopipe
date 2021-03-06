import os
import logging
import re
import functools

from matplotlib import pyplot as plt

from . import mpi
from . import utils

logger = logging.getLogger('Plotting')


class BasePlotStyle(utils.BaseClass):

    logger = logging.getLogger('BasePlotStyle')

    @mpi.MPIInit
    def __init__(self, style=None, **kwargs):
        if isinstance(style, self.__class__):
            self.__dict__.update(style.__dict__)
            self.update(**kwargs)
            return
        self.kwfig = {}
        self.update(**kwargs)

    def update(self, **kwargs):
        for key,val in kwargs.items():
            setattr(self,key,val)

    def savefig(self, filename, fig=None, root=0):
        if self.is_mpi_root():
            savefig(filename,fig=fig,**self.kwfig)


def savefig(filename, fig=None, bbox_inches='tight', pad_inches=0.1, dpi=200, **kwargs):
    """Save matplotlib figure to ``filename``."""
    utils.mkdir(os.path.dirname(filename))
    logger.info('Saving figure to {}.'.format(filename))
    if fig is None:
        fig = plt.gcf()
    fig.savefig(filename,bbox_inches=bbox_inches,pad_inches=pad_inches,dpi=dpi,**kwargs)
    plt.close(fig)


def suplabel(axis,label,shift=0,labelpad=5,ha='center',va='center',**kwargs):
    """
    Add super ylabel or xlabel to the figure. Similar to matplotlib.suptitle.
    Taken from https://stackoverflow.com/questions/6963035/pyplot-axes-labels-for-subplots.

    Parameters
    ----------
    axis : str
        'x' or 'y'.
    label : str
        label.
    shift : float, optional
        shift.
    labelpad : float, optional
        padding from the axis.
    ha : str, optional
        horizontal alignment.
    va : str, optional
        vertical alignment.
    kwargs : dict
        kwargs for :meth:`matplotlib.pyplot.text`
    """
    fig = plt.gcf()
    xmin = []
    ymin = []
    for ax in fig.axes:
        xmin.append(ax.get_position().xmin)
        ymin.append(ax.get_position().ymin)
    xmin,ymin = min(xmin),min(ymin)
    dpi = fig.dpi
    if axis.lower() == 'y':
        rotation = 90.
        x = xmin - float(labelpad)/dpi
        y = 0.5 + shift
    elif axis.lower() == 'x':
        rotation = 0.
        x = 0.5 + shift
        y = ymin - float(labelpad)/dpi
    else:
        raise Exception('Unexpected axis {}; chose between x and y'.format(axis))
    plt.text(x,y,label,rotation=rotation,transform=fig.transFigure,ha=ha,va=va,**kwargs)
