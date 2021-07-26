import numpy as np

from cosmopipe import section_names
from cosmopipe.lib.data_vector import DataVector
from cosmopipe.lib import utils

from .data_vector import get_kwview, update_data_projs


def _cast_to_arrays(x, n=1):
    if x is not None and (isinstance(x,dict) or np.ndim(x[0]) == 0):
        x = [x]*n
    if isinstance(x,list):
        if isinstance(x[0],dict):
            x = [utils.customspace(**kw) for kw in x]
    return x


class SyntheticDataVector(object):

    def setup(self):
        xin = self.options.get('x',None)
        y = self.options.get('y',None)
        projs = self.options.get('projs',None)
        edges = self.options.get('edges',None)
        data_load = self.options.get('data_load',False)
        save = self.options.get('save',False)

        if data_load:
            if isinstance(data_load,bool) and data_load:
                data_load = 'data_vector'
            data_vector = self.data_block[syntax.split_sections(data_load,default_section=section_names.data)]

        if y is not None: y = _cast_to_arrays(y,n=len(projs))
        if edges is not None:
            edges = _cast_to_arrays(edges,n=len(projs))

        if xin is None:
            if edges is None:
                raise ValueError('Either x or edges must be provided')
            xin = [(edge[:-1] + edge[1:])/2. for edge in edges]
        if isinstance(xin,dict) or np.ndim(xin[0]) == 0:
            xin = [xin]*len(projs)
        x = []
        for x_,proj in zip(xin,projs):
            kwargs = x_.copy()
            if isinstance(x_,dict):
                if x_.get('min',None) is None:
                    if not data_load: raise ValueError('min must be specified')
                    kwargs['min'] = data_vector.get_x(proj=proj).min()
                if x_.get('max',None) is None:
                    if not data_load: raise ValueError('max must be specified')
                    kwargs['max'] = data_vector.get_x(proj=proj).max()
                x.append(utils.customspace(**kwargs))
            else:
                x.append(x_)
        if data_load:
            if isinstance(data_load,bool) and data_load:
                data_load = 'data_vector'
            data_vector = self.data_block[syntax.split_sections(data_load,default_section=section_names.data)]
            self.data_vector = data_vector.copy(copy_proj=True)
            if projs is not None:
                self.data_vector.data = [self.data_vector.get(proj).copy() for proj in projs]
            if x is not None:
                for iproj,dataproj in enumerate(self.data_vector.data): dataproj.set_x(x[iproj])
            if y is not None:
                self.data_vector.set_y(y,concatenated=False)
            if edges is not None:
                for proj,edges in zip(projs,edges):
                    dataproj = self.data_vector.get(proj)
                    dataproj.edges[dataproj.attrs['x']] = edges
            #data = data.noview() # required for now, because covariance matrix without view should have data vectors without view
        else:
            self.data_vector = DataVector(x=x,y=y,proj=projs,edges=[{'x':edge} for edge in edges] if edges is not None else None)
        #print(id(self.data_vector.projs[0]))
        update_data_projs(self.data_vector.projs,self.options.get('projs_attrs',[]))
        #print(id(self.data_vector.projs[0]))
        if save: self.data_vector.save_auto(save)
        kwview = get_kwview(self.data_vector,xlim=self.options.get('xlim',None))
        self.data_vector = self.data_vector.view(**kwview)

        data_vector = self.data_block.get(section_names.data,'data_vector',[])
        data_vector += self.data_vector
        self.data_block[section_names.data,'data_vector'] = data_vector
        if y is not None: self.data_block[section_names.data,'y'] = data_vector.get_y()

    def execute(self):
        pass

    def cleanup(self):
        pass
