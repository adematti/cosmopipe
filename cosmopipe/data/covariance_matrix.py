import os

from pypescript import SectionBlock, syntax

from cosmopipe import section_names
from cosmopipe.lib import syntax
from cosmopipe.lib.data import CovarianceMatrix


def setup(name, config_block, data_block):
    options = SectionBlock(config_block,name)
    covariance_load = options.get_string('covariance_load','covariance_matrix')
    data = options.get('data',None) or {}
    if os.sep in covariance_load:
        if os.path.splitext(covariance_load)[-1]:
            kwargs = {'xdim':None,'comments':'#','usecols':None,'skip_rows':0,'max_rows':None,'proj':False,'mapping_header':None,'mapping_proj':None,'type':None}
            for name,value in kwargs.items():
                kwargs[name] = options.get(name,value)
            if kwargs['type'] is None: del kwargs['type']
            if data:
                from .data_vector import get_data_from_options
                kwargs['data'] = get_data_from_options(self.options['data'],data_block=data_block)
            cov = CovarianceMatrix.load_txt(covariance_load,**kwargs)
        else:
            cov = CovarianceMatrix.load(covariance_load)
    else:
        key = syntax.split_sections(covariance_load,default_section=section_names.data)
        cov = data_block[key]
    #from .data_vector import get_kwview
    #projs = get_kwview(cov.x[0],xlim=data.get('xlim'))

    projs = data_block[section_names.data,'projs']
    xlims = data_block[section_names.data,'xlims']
    cov = cov.view(proj=projs,xlim=xlims)

    data_block[section_names.covariance,'covariance_matrix'] = cov
    data_block[section_names.covariance,'cov'] = cov.get_cov()
    data_block[section_names.covariance,'invcov'] = cov.get_invcov()
    data_block[section_names.covariance,'nobs'] = cov.attrs.get('nobs',None)


def execute(name, config_block, data_block):
    pass


def cleanup(name, config_block, data_block):
    pass
