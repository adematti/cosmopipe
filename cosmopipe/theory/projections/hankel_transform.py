import numpy as np
from scipy import interpolate

from pypescript.config import ConfigError

from cosmopipe import section_names
from cosmopipe.lib import utils
from cosmopipe.lib.utils import BaseClass
from cosmopipe.lib.theory.base import ModelCollection
from cosmopipe.lib.theory import hankel_transform
from cosmopipe.lib.theory.projection import ProjectionBase


class HankelTransform(object):

    def setup(self):
        options = {'nx':1024,'q':1.5,'ells':(0,2,4),'integration':None}
        for name,value in options.items():
            options[name] = self.options.get(name,value)
        model_names = self.options.get_list('model_names',None)
        model_attrs = [{'space':ProjectionBase.POWER},{'space':ProjectionBase.CORRELATION}]
        if model_names is not None:
            model_attrs['name'] = set(model_names)
        origin_collection = self.data_block[section_names.model,'collection']
        tmp = origin_collection.select(model_attrs)
        collection = ModelCollection()
        for base,model in tmp:
            if not isinstance(model,hankel_transform.HankelTransform):
                collection.set(model,base=base)
        if len(collection) > 1 and model_names is None:
            raise ConfigError('Found several models that can be Hankel-transformed: {}; specify the one of interest with "model_names"'.format(collection))

        self.collection = ModelCollection()
        for base,model in collection:
            ht = hankel_transform.HankelTransform(model=model,base=base,**options)
            self.collection.set(ht)

        self.data_block[section_names.model,'collection'] = origin_collection + self.collection

    def execute(self):
        for base,model in self.collection:
            model.input_model = self.data_block[section_names.model,'collection'].get(model.input_base)
        self.data_block[section_names.model,'collection'] = self.data_block.get(section_names.model,'collection',{}) + self.collection

    def cleanup(self):
        pass
