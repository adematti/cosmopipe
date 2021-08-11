"""Some plotting utilities."""

import os
import re
import logging
import functools

import numpy as np
from matplotlib import pyplot as plt

from . import mpi
from . import utils

logger = logging.getLogger('Plotting')


def make_list(obj, length=1):
    """
    Return list from ``obj``.

    Parameters
    ----------
    obj : object, tuple, list, array
        If tuple, list or array, cast to list.
        Else return list of ``obj`` with length ``length``.

    length : int, default=1
        Length of list to return, if ``obj`` not already tuple, list or array.

    Returns
    -------
    toret : list
    """
    if isinstance(obj,tuple):
        return list(obj)
    if isinstance(obj,np.ndarray):
        return obj.tolist()
    if not isinstance(obj,list):
        return [obj for i in range(length)]
    return obj


class BasePlotStyle(utils.BaseClass):
    """
    Base class to represent a plotting style.
    It holds attributes that can be set at initialization (``style = BaseDataPlotStyle(color='r')``) or at any
    time using :meth:`update`.
    """
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
        """Update attibutes with those in ``kwargs``."""
        for key,val in kwargs.items():
            setattr(self,key,val)

    def savefig(self, filename, fig=None):
        """
        Save figure to ``filename``.

        Parameters
        ----------
        filename : string
            Path where to save figure.

        fig : matplotlib.figure.Figure, default=None
            Figure to save. Defaults to current figure.
        """
        if self.is_mpi_root():
            savefig(filename,fig=fig,**self.kwfig)

    def get(self, name, value=None, default=None):
        """
        Return ``value`` if not ``None``, else attribute ``name`` if not ``None``,
        else ``default``.

        Parameters
        ----------
        name : string
            Attribute name. If ``None``, defaults to ``default``.

        value : object, default=None
            Value. If ``None``, returns attribute ``name``.

        default : object, default=None
            Default value.
        """
        if value is not None:
            return value
        value = getattr(self,name,None)
        if value is None:
            return default
        return value

    def get_list(self, name, value=None, default=None):
        """
        Same as :meth:`get`, but ensuring returned value is a list.
        Default length (see :func:`make_list`) is taken as ``default`` length.
        """
        if value is not None:
            return make_list(value,length=len(default) if default is not None else 1)
        value = getattr(self,name,None)
        if value is None:
            return default
        return make_list(value,length=len(default) if default is not None else 1)


def savefig(filename, fig=None, bbox_inches='tight', pad_inches=0.1, dpi=200, **kwargs):
    """
    Save figure to ``filename``.

    Parameters
    ----------
    filename : string
        Path where to save figure.

    fig : matplotlib.figure.Figure, default=None
        Figure to save. Defaults to current figure.

    kwargs : dict
        Arguments for :meth:`matplotlib.figure.Figure.savefig`.
    """
    utils.mkdir(os.path.dirname(filename))
    logger.info('Saving figure to {}.'.format(filename))
    if fig is None:
        fig = plt.gcf()
    fig.savefig(filename,bbox_inches=bbox_inches,pad_inches=pad_inches,dpi=dpi,**kwargs)
    plt.close(fig)


def suplabel(axis, label, shift=0, labelpad=5, ha='center', va='center', **kwargs):
    """
    Add global x-coordinate or y-coordinate label to the figure. Similar to matplotlib.suptitle.
    Taken from https://stackoverflow.com/questions/6963035/pyplot-axes-labels-for-subplots.

    Parameters
    ----------
    axis : str
        'x' or 'y'.

    label : string
        Label string.

    shift : float, optional
        Shift along ``axis``.

    labelpad : float, optional
        Padding perpendicular to ``axis``.

    ha : str, optional
        Label horizontal alignment.

    va : str, optional
        Label vertical alignment.

    kwargs : dict
        Arguments for :func:`matplotlib.pyplot.text`.
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
