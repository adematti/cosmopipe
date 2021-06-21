import os

import numpy as np
from nbodykit import setup_logging
from nbodykit.lab import *

from cosmopipe.lib.catalog import Catalog, RandomCatalog
from cosmopipe.lib.catalog import utils


def generate_lognormal():
    base_dir = '_catalog'
    
    redshift = 1.
    cosmo = cosmology.Planck15
    Plin = cosmology.LinearPower(cosmo, redshift, transfer='CLASS')
    BoxSize = 1380.
    nbar = 3e-4
    seed = 42
    cat = LogNormalCatalog(Plin=Plin, nbar=nbar, BoxSize=BoxSize, Nmesh=256, bias=2.0, seed=seed)
    cat = Catalog.from_nbodykit(cat)
    fn = os.path.join(base_dir,'lognormal_box.fits')
    cat.save_fits(fn)

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

    fn = os.path.join(base_dir,'lognormal_data.fits')
    cat.save_fits(fn)
    cat = RandomCatalog(BoxSize=BoxSize,BoxCenter=cat.attrs['BoxSize']/2.,nbar=2*nbar,seed=seed,mpistate='scattered')
    distance_rsd,cat['RA'],cat['DEC'] = utils.cartesian_to_sky(cat['Position'])
    cat['Z'] = distance_to_redshift(distance_rsd)
    cat['NZ'] = cat.ones()*nbar
    fn = os.path.join(base_dir,'lognormal_randoms.fits')
    cat.save_fits(fn)


if __name__ == '__main__':

    setup_logging()
    generate_lognormal()
