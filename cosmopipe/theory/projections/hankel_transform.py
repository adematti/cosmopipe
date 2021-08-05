import numpy as np
from scipy import interpolate

from pypescript.config import ConfigError

from cosmopipe import section_names
from cosmopipe.lib import utils
from cosmopipe.lib.theory.base import ProjectionBase, ModelCollection
from cosmopipe.lib.theory import hankel_transform


class HankelTransform(object):

    def setup(self):
        options = dict(nx=1024,q=0,ells=(0,2,4),integration=None)
        for name,value in options.items():
            options[name] = self.options.get(name,value)
        model_names = self.options.get_list('model_names',None)
        model_attrs = [{'space':ProjectionBase.POWER},{'space':ProjectionBase.CORRELATION}]
        if model_names is not None:
            model_attrs['name'] = set(model_names)
        self.model_collection = self.data_block[section_names.model,'collection']
        tmp = self.model_collection.select(*model_attrs)
        collection = ModelCollection()
        for model in tmp:
            if not isinstance(model,hankel_transform.HankelTransform):
                collection.set(model)
        if len(collection) > 1 and model_names is None:
            raise ConfigError('Found several models that can be Hankel-transformed: {}; specify the one(s) of interest with "model_names"'.format(collection))

        hankel_collection = ModelCollection()
        for model in collection:
            ht = hankel_transform.HankelTransform(model=model,**options)
            hankel_collection.set(ht)

        self.model_collection += hankel_collection

    def execute(self):
        pass

    def cleanup(self):
        pass
