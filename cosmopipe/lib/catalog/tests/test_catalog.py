import os

import numpy as np
import fitsio

from cosmopipe.lib import setup_logging
from cosmopipe.lib.catalog import Catalog


fits_fn = os.path.join(os.path.dirname(__file__),'test.fits')


def make_fits_catalog():
    size = 100
    array = np.zeros(size,dtype=[('RA','f8'),('DEC','f8'),('Z','f8'),('Position','f8',3)])
    header = [dict(name='answer',value='42',comment='COMMENT')]
    fitsio.write(fits_fn,array,header=header,clobber=True)


def test_catalog():
    catalog = Catalog.load_auto(fits_fn)
    catalog.save_auto(fits_fn)
    catalog2 = Catalog.load_auto(fits_fn)
    assert catalog2 == catalog


if __name__ == '__main__':

    setup_logging()
    #make_fits_catalog()
    test_catalog()
