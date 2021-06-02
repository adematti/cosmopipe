import os
import numpy as np

from pypescript import SectionBlock, syntax

from cosmopipe import section_names
from cosmopipe.lib.data import DataVector


def get_data_from_options(options):
    data_load = options.get_string('data_load',None)
    if data_load is not None:
        if os.path.splitext(data_load)[-1]:
            kwargs = {'xdim':options.get_int('xdim',None),'comments':options.get_string('comments','#'),'usecols':options.get_list('usecols',None)}
            kwargs.update({'skip_rows':options.get_int('skip_rows',0),'max_rows':options.get_int('max_rows',None)})
            kwargs.update({'proj':options.get_bool('col_proj',False),'mapping_header':options.get('mapping_header',None),'mapping_proj':options.get('mapping_proj',None)})
            data = DataVector.load_txt(data_load,**kwargs)
        else:
            data = DataVector.load(data_load)
    else:
        data_key = syntax.split_sections(options.get('data_key','data_vector'),default_section=section_names.data)
        data = data_block[data_key]
    return data


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
    data_block[section_names.data,'xlims'] = np.array(data.kwview['xlim'])
    data_block[section_names.data,'projs'] = np.array(data.kwview['proj'])
    data_block[section_names.data,'data_vector'] = data
    data_block[section_names.data,'x'] = data.get_x()
    data_block[section_names.data,'y'] = data.get_y()
    data_block[section_names.data,'shotnoise'] = data.attrs.get('shotnoise',0.)


def execute(name, config_block, data_block):
    pass


def cleanup(name, config_block, data_block):
    pass
