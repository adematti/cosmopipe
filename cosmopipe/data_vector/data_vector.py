import os
import numpy as np

from cosmopipe import section_names
from cosmopipe.lib import syntax, data_vector


def get_data_from_options(options, data_load, data_block=None, default_section=section_names.data, loader=data_vector.DataVector.load_auto):
    kwargs = dict(comments='#',usecols=None,skip_rows=0,max_rows=None,mapping_header=None,mapping_proj=None,attrs=None)
    for name,value in kwargs.items():
        kwargs[name] = options.get(name,value)
    data = syntax.load_auto(data_load,data_block=data_block,default_section=default_section,loader=loader,squeeze=True,**kwargs)
    projs = options.get('projs',{})
    for projname in projs:
        for proj in data.projs:
            if proj.name == projname:
                proj.set(**projs[projname])
    apply = options.get('apply',[])
    for apply_ in apply:
        for key,value in apply_.items():
            value = value.copy()
            projs = value.pop('projs',data.projs)
            if not isinstance(projs,list):
                projs = [projs]
            for proj in projs:
                toret = getattr(data.get(proj),key)(**value)
                if isinstance(toret,list):
                    for dataproj in toret:
                        data.set(dataproj)
                elif toret is not None:
                    data.set(toret)
    return data


def get_kwview(data, xlim=None):
    projs = data.projs
    if xlim is not None:
        if not isinstance(xlim,list):
            projs = list(xlim.keys())
            xlim = [xlim[proj] for proj in projs]
    else:
        xlim = [[-np.inf,np.inf] for proj in projs]
    return dict(proj=projs,xlim=xlim)


class DataVector(object):

    def setup(self):
        self.data_vector = get_data_from_options(self.options,data_load=self.options['data_load'],data_block=self.data_block)
        self.data_vector = self.data_vector.view(**get_kwview(self.data_vector,xlim=self.options.get('xlim',None)))
        data_vector = self.data_block.get(section_names.data,'data_vector',[])
        data_vector += self.data_vector
        self.data_block[section_names.data,'data_vector'] = data_vector
        self.data_block[section_names.data,'y'] = data_vector.get_y()
        #print(self.data.get_x())

    def execute(self):
        pass

    def cleanup(self):
        pass
