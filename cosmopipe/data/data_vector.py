import os
import numpy as np

from pypescript import SectionBlock

from cosmopipe import section_names
from cosmopipe.lib.data import DataVector


def get_data_from_options(options):
    data_load = options['data_load']
    kwargs = {'xdim':options.get_int('xdim',None),'comments':options.get_string('comments','#'),'usecols':options.get_list('usecols',None)}
    kwargs.update({'skip_rows':options.get_int('skip_rows',0),'max_rows':options.get_int('max_rows',None)})
    kwargs.update({'proj':options.get_bool('col_proj',False),'mapping_header':options.get('mapping_header',None),'mapping_proj':options.get('mapping_proj',None)})
    t = options.get_string('type',None)
    if t is not None: kwargs['type'] = t
    return syntax.load_auto(data_load,data_block=data_block,default_section=section_names.data,loader=DataVector.load_auto,squeeze=True,**kwargs)

def get_kwview(data, xlim=None):
    projs = data.projs
    if xlim is not None:
        if not isinstance(xlim,list):
            projs = list(xlim.keys())
            xlims = [xlim[proj] for proj in projs]
    else:
        xlims = [[-np.inf,np.inf] for proj in projs]
    return dict(proj=projs,xlim=xlims)


def setup(name, config_block, data_block):
    options = SectionBlock(config_block,name)
    data = get_data_from_options(options)
    data = data.view(**get_kwview(data,xlim=options.get('xlim',None)))
    data_block[section_names.data,'xlims'] = data.kwview['xlim']
    data_block[section_names.data,'projs'] = data.kwview['proj']
    data_block[section_names.data,'data_vector'] = data
    data_block[section_names.data,'x'] = data.get_x()
    data_block[section_names.data,'y'] = data.get_y()
    data_block[section_names.data,'shotnoise'] = data.attrs.get('shotnoise',0.)


def execute(name, config_block, data_block):
    pass


def cleanup(name, config_block, data_block):
    pass
