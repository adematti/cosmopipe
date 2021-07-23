import os

import numpy as np
from nbodykit import setup_logging
from nbodykit.lab import *

from cosmopipe.lib.catalog import Catalog, RandomCatalog
from cosmopipe.lib.catalog import utils


def generate_lognormal(data_fn, randoms_fn=None, data_box_fn=None, seed=42):

    redshift = 1.
    cosmo = cosmology.Planck15
    Plin = cosmology.LinearPower(cosmo, redshift, transfer='CLASS')
    BoxSize = 1380.
    nbar = 3e-4
    cat = LogNormalCatalog(Plin=Plin, nbar=nbar, BoxSize=BoxSize, Nmesh=256, bias=2.0, seed=seed)
    cat = Catalog.from_nbodykit(cat)

    if data_box_fn:
        cat.save_fits(data_box_fn)

    distance,ra,dec = utils.cartesian_to_sky(cat['Position'])
    los = cat['Position']/distance[:,None]
    distance_to_redshift = utils.DistanceToRedshift(cosmo.comoving_distance)
    cat['Z_COSMO'] = distance_to_redshift(distance)
    tmp = cat['Position'] + utils.vector_projection(cat['VelocityOffset'],cat['Position'])
    distance_rsd,cat['RA'],cat['DEC'] = utils.cartesian_to_sky(cat['Position'] + utils.vector_projection(cat['VelocityOffset'],cat['Position']))
    #assert np.allclose(cat['RA'],ra) and np.allclose(cat['DEC'],dec)
    cat['Z'] = distance_to_redshift(distance_rsd)
    cat['DZ'] = cat['Z'] - cat['Z_COSMO']
    cat['NZ'] = cat.ones()*nbar

    if data_fn is not None:
        cat.save_fits(data_fn)

    if randoms_fn is not None:
        cat = RandomCatalog(BoxSize=BoxSize,BoxCenter=cat.attrs['BoxSize']/2.,nbar=2*nbar,seed=seed,mpistate='scattered')
        distance_rsd,cat['RA'],cat['DEC'] = utils.cartesian_to_sky(cat['Position'])
        cat['Z'] = distance_to_redshift(distance_rsd)
        cat['NZ'] = cat.ones()*nbar
        cat.save_fits(randoms_fn)


if __name__ == '__main__':

    setup_logging()
    base_dir = '_catalog'
    data_fn = os.path.join(base_dir,'lognormal_data.fits')
    data_box_fn = os.path.join(base_dir,'lognormal_box.fits')
    randoms_fn = os.path.join(base_dir,'lognormal_randoms.fits')
    generate_lognormal(data_fn,randoms_fn=randoms_fn,data_box_fn=data_box_fn,seed=42)
    for ii in range(1,11):
        data_fn = os.path.join(base_dir,'lognormal_data_{:d}.fits'.format(ii))
        generate_lognormal(data_fn,seed=ii)
