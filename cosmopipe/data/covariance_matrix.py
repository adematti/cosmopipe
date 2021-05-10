from pypescript import SectionBlock

from cosmopipe import section_names
from cosmopipe.lib.data import CovarianceMatrix


def setup(name, config_block, data_block):
    options = SectionBlock(config_block,name)
    if options.get_string('format','txt') == 'txt':
        kwargs = {'xdim':options.get_int('xdim',None),'comments':options.get_string('comments','#'),'usecols':options.get_list('usecols',None)}
        kwargs.update({'skip_rows':options.get_int('skip_rows',0),'max_rows':options.get_int('max_rows',None)})
        kwargs.update({'proj':options.get_bool('col_proj',False),'mapping_header':options.get('mapping_header',None),'mapping_proj':options.get('mapping_proj',None)})
        if options.has('data'):
            from .data_vector import get_data_from_options
            kwargs['data'] = get_data_from_options(SectionBlock(config_block,options.get_string('data')))
        cov = CovarianceMatrix.load_txt(options.get_string('covariance_file'),**kwargs)
    else:
        cov = CovarianceMatrix.load(options.get_string('covariance_file'))

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
