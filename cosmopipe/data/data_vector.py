import numpy as np

from pypescript import SectionBlock

from cosmopipe import section_names
from cosmopipe.lib.data import DataVector


def get_data_from_options(options):
    data_file = options.get_string('data_file')
    if data_file.split('.')[-1] == 'txt':
        kwargs = {'xdim':options.get_int('xdim',None),'comments':options.get_string('comments','#'),'usecols':options.get_list('usecols',None)}
        kwargs.update({'skip_rows':options.get_int('skip_rows',0),'max_rows':options.get_int('max_rows',None)})
        kwargs.update({'proj':options.get_bool('col_proj',False),'mapping_header':options.get('mapping_header',None),'mapping_proj':options.get('mapping_proj',None)})
        data = DataVector.load_txt(data_file,**kwargs)
    else:
        data = DataVector.load(data_file)
    return data


def get_kwview(data, options):
    projs = data.projs
    if options.has('xlim'):
        xlims = options['xlim']
        if not isinstance(xlims,list):
            projs = list(xlims.keys())
            xlims = [xlims[proj] for proj in projs]
    else:
        xlims = [[-np.inf,np.inf] for proj in projs]
    return dict(proj=projs,xlim=xlims)


def setup(name, config_block, data_block):
    options = SectionBlock(config_block,name)
    data = get_data_from_options(options)
    data = data.view(**get_kwview(data,options))
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
