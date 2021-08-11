import os

from pypescript import SectionBlock, syntax

from cosmopipe import section_names
from cosmopipe.lib import syntax, data_vector

from .data_vector import update_data_projs


class CovarianceMatrix(object):

    def setup(self):
        covariance_load = self.options.get_string('covariance_load','covariance_matrix')
        data = self.options.get('data',None) or {}
        if os.sep in covariance_load:
            if os.path.splitext(covariance_load)[-1]:
                kwargs = dict(mapping_header=None,mapping_proj=None,comments='#',usecols=None,columns=None,skip_rows=0,max_rows=None,attrs=None)
                for name,value in kwargs.items():
                    kwargs[name] = self.options.get(name,value)
                if data:
                    from .data_vector import get_data_from_options
                    kwargs['data'] = get_data_from_options(self.options['data'],data_load=self.options['data']['data_load'],data_block=data_block)
                cov = data_vector.CovarianceMatrix.load_txt(covariance_load,**kwargs)
            else:
                cov = data_vector.CovarianceMatrix.load(covariance_load)
        else:
            key = syntax.split_sections(covariance_load,default_section=section_names.data)
            cov = self.data_block[key]

        for x in cov.x:
            update_data_projs(x.projs,self.options.get('projs_attrs',[]))

        apply = self.options.get('apply',[])
        for apply_ in apply:
            for key,value in apply_.items():
                getattr(cov,key)(**value)

        kwview = {}
        if self.data_block.has(section_names.data,'data_vector'):
            kwview = self.data_block[section_names.data,'data_vector'].kwview
        cov = cov.view(**kwview)
        self.data_block[section_names.covariance,'covariance_matrix'] = cov
        self.data_block[section_names.covariance,'cov'] = cov.get_cov()
        self.data_block[section_names.covariance,'invcov'] = cov.get_invcov()
        self.data_block[section_names.covariance,'nobs'] = cov.attrs.get('nobs',None)

    def execute(self):
        pass

    def cleanup(self):
        pass
