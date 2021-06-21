import logging

import numpy as np
from nbodykit.lab import FFTPower
from pypescript import BaseModule

from cosmopipe import section_names
from cosmopipe.lib import syntax
from cosmopipe.lib.catalog import Catalog
from cosmopipe.lib.data import DataVector


class LogNormalCatalog(BaseModule):

    logger = logging.getLogger('LogNormalCatalog')

    def setup(self):
        self.redshift = self.options.get('redshift',1.)
        self.catalog_options = {'nbar':3e-4,'BoxSize':1000.,'Nmesh':256,'bias':2.0,'seed':None}
        for name,value in self.catalog_options.items():
            self.catalog_options[name] = self.options.get(name,value)


    def execute(self):

        cosmo = self.data_block.get(section_names.fiducial_cosmology,'cosmo',None)

        cosmo = cosmology.Planck15
        Plin = cosmology.LinearPower(cosmo, redshift, transfer='CLASS')
        BoxSize = 1380.
        nbar = 3e-4
        seed = 42
        cat = LogNormalCatalog(Plin=Plin,**self.catalog_options)
        cat = Catalog.from_nbodykit(cat)
        fn = os.path.join(base_dir,'lognormal_box.fits')
        cat.save_fits(fn)
