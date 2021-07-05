import os
import numpy as np

from cosmopipe import section_names
from cosmopipe.lib import syntax, data_vector


def get_data_from_options(options, data_block=None):
    data_load = options['data_load']
    kwargs = dict(comments='#',usecols=None,skip_rows=0,max_rows=None,mapping_header=None,mapping_proj=None,attrs=None)
    for name,value in kwargs.items():
        kwargs[name] = options.get(name,value)
    data = syntax.load_auto(data_load,data_block=data_block,default_section=section_names.data,loader=data_vector.DataVector.load_auto,squeeze=True,**kwargs)
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
        data = get_data_from_options(self.options,data_block=self.data_block)
        self.data = data.view(**get_kwview(data,xlim=self.options.get('xlim',None)))
        self.data_block[section_names.data,'data_vector'] = self.data_block.get(section_names.data,'data_vector',[]) + self.data
        self.data_block[section_names.data,'y'] = self.data_block[section_names.data,'data_vector'].get_y()

    def execute(self):
        pass

    def cleanup(self):
        pass