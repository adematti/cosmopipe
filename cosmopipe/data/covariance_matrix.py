import os

from pypescript import SectionBlock, syntax

from cosmopipe import section_names
from cosmopipe.lib.data import CovarianceMatrix


def setup(name, config_block, data_block):
    options = SectionBlock(config_block,name)
    covariance_load = options.get_string('covariance_load',None)
    if covariance_load is not None:
        if os.path.splitext(covariance_load)[-1]:
            kwargs = {'xdim':options.get_int('xdim',None),'comments':options.get_string('comments','#'),'usecols':options.get_list('usecols',None)}
            kwargs.update({'skip_rows':options.get_int('skip_rows',0),'max_rows':options.get_int('max_rows',None)})
            kwargs.update({'proj':options.get_bool('col_proj',False),'mapping_header':options.get('mapping_header',None),'mapping_proj':options.get('mapping_proj',None)})
            if options.has('data'):
                from .data_vector import get_data_from_options
                kwargs['data'] = get_data_from_options(self.options['data'])
            cov = CovarianceMatrix.load_txt(covariance_load,**kwargs)
        else:
            cov = CovarianceMatrix.load(covariance_load)
    else:
        key = syntax.split_sections(options.get('covariance_key','covariance_matrix'),default_section=section_names.data)
        cov = data_block[key]
    from .data_vector import get_kwview

    projs = get_kwview(cov.x[0],options.get('data',{}).get(''))

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
