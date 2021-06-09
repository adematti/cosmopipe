import os
from pypescript.syntax import *


def load_auto(loads, data_block=None, default_section=None, loader=None, squeeze=False, **kwargs):
    """Load either from data_block or file"""
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
