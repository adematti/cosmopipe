import os

import numpy as np
from nbodykit import setup_logging
from nbodykit.lab import *

from cosmopipe.lib.catalog import Catalog, RandomBoxCatalog
from cosmopipe.lib.catalog import utils


def generate_lognormal(data_fn, randoms_fn=None, data_box_fn=None, seed=42,use_existing=False):
    if use_existing :
        if (not data_fn or os.path.isfile(data_fn)) and \
           (not randoms_fn or os.path.isfile(randoms_fn)) and \
           (not data_box_fn or os.path.isfile(data_box_fn)) : 
            return 

    redshift = 1.
    cosmo = cosmology.Planck15
    Plin = cosmology.LinearPower(cosmo, redshift, transfer='CLASS')
    BoxSize = 1380.
    nbar = 3e-4
    cat = LogNormalCatalog(Plin=Plin, nbar=nbar, BoxSize=BoxSize, Nmesh=256, bias=2.0, seed=seed)
    cat = Catalog.from_nbodykit(cat)


    if data_box_fn:
        cat.save_fits(data_box_fn)

    offset = cosmo.comoving_distance(redshift) - BoxSize/2.
    cat['Position'][:,0] += offset
    distance,ra,dec = utils.cartesian_to_sky(cat['Position'])
    los = cat['Position']/distance[:,None]
    distance_to_redshift = utils.DistanceToRedshift(cosmo.comoving_distance)
    cat['Z_COSMO'] = distance_to_redshift(distance)
    distance_rsd,cat['RA'],cat['DEC'] = utils.cartesian_to_sky(cat['Position'] + utils.vector_projection(cat['VelocityOffset'],cat['Position']))
    #assert np.allclose(cat['RA'],ra) and np.allclose(cat['DEC'],dec)
    cat['Z'] = distance_to_redshift(distance_rsd)
    cat['DZ'] = cat['Z'] - cat['Z_COSMO']
    nz = cat.gsize*1./BoxSize**3
    cat['NZ'] = cat.ones()*nz

    if data_fn is not None:
        cat.save_fits(data_fn)

    if randoms_fn is not None:
        cat = RandomBoxCatalog(BoxSize=BoxSize,BoxCenter=cat.attrs['BoxSize']/2.,nbar=2*nbar,seed=seed,mpistate='scattered')
        cat['Position'][:,0] += offset
        distance_rsd,cat['RA'],cat['DEC'] = utils.cartesian_to_sky(cat['Position'])
        cat['Z'] = distance_to_redshift(distance_rsd)
        cat['NZ'] = cat.ones()*nz
        cat.save_fits(randoms_fn)


def main(ndata=11,use_existing=False) :
    setup_logging()
    base_dir = '_catalog'
    data_box_fn = os.path.join(base_dir,'lognormal_box.fits')
    data_fn = os.path.join(base_dir,'lognormal_data.fits')
    randoms_fn = os.path.join(base_dir,'lognormal_randoms.fits')
    generate_lognormal(data_fn,randoms_fn=randoms_fn,data_box_fn=data_box_fn,seed=42,use_existing=use_existing)
    for ii in range(1,ndata):
        data_fn = os.path.join(base_dir,'lognormal_data_{:d}.fits'.format(ii))
        generate_lognormal(data_fn,seed=ii,use_existing=use_existing)

if __name__ == '__main__':
    main()

