"""**pypescript**-related convenience functions."""

import os
from pypescript.syntax import *


def load_auto(loads, data_block=None, default_section=None, loader=None, squeeze=False, **kwargs):
    r"""
    Load either from data_block or file.

    Parameters
    ----------
    loads : string, tuple, list
        Can be:

        - name of file to load, in which case ``loads`` must contain :attr:`os.sep`, e.g. '\'
        - section.name or name (prepended by ``default_section``) in ``data_block``
        - any list or tuple of the above two options

    data_block : :class:`pypescript.DataBlock`, default=None
        Current data block, used if ``loads`` does not refer to a file name.

    default_section : string, default=None
        Default section where to find ``loads`` if the latter refers only to a name in ``data_block``.

    loader : callable, default=None
        Function to load file from disk.

    squeeze : bool, default=False
        If ``True`` and ``loads`` is string, or tuple and list with one element, return single object.

    kwargs : dict
        Arguments for ``loader``.
    """
    isscalar = not isinstance(loads,(tuple,list))
    if isscalar: loads = [loads]
    toret = []
    for load in loads:
        if load is not None and os.sep in load:
            toret.append(loader(load,**kwargs))
        else:
            key = split_sections(load,default_section=default_section)
            toret.append(data_block[key])
    if squeeze and len(toret) == 1:
        toret = toret[0]
    return toret
