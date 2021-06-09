def prepare_survey_angular_catalogs(data, randoms, ra='RA', dec='DEC', weight_comp=None):

    origin_catalogs = {'data':data,'randoms':randoms}
    catalogs = {name:catalog.copy(columns=[]) for name,catalog in origin_catalogs.items()}

    def from_origin(column):
        for name,catalog in catalogs.items():
            catalog[name] = origin_catalogs.eval(self.catalog_options[name])

    if weight_comp is None:
        for name,catalog in catalogs.items():
            catalog['weight_comp'] = origin_catalogs[name].ones()
    else:
        from_origin('weight_comp')

    from_origin('ra')
    from_origin('dec')

    return catalogs['data'],catalogs['randoms']


def prepare_survey_catalogs(data, randoms, cosmo=None, ra='RA', dec='DEC', z='Z', weight_comp=None, nbar='NZ', weight_fkp=None, P0_fkp=0.):

    origin_catalogs = {'data':data,'randoms':randoms}

    catalogs = {}
    catalogs['data'],catalogs['randoms'] = prepare_survey_angular_catalogs(data,randoms,ra=ra,dec=dec,weight_comp=weiht_comp)

    def from_origin(column):
        for name,catalog in catalogs.items():
            catalog[name] = origin_catalogs.eval(self.catalog_options[name])

    if z is not None:
        from_origin('z')

    if self.catalog_options['position'] is None:
        cosmo = self.data_block[section_names.fiducial_cosmology,'cosmo']
        for name,catalog in catalogs.items():
            catalog['distance'] = cosmo.comoving_radial_distance(catalog['z'])
            catalog['position'] = utils.sky_to_cartesian(distance,catalog['ra'],catalog['dec'],degree=True)
    else:
        from_origin('position')
        catalog['distance'] = utils.distance(catalog['position'])

    if isinstance(nbar,dict):
        if 'z' in randoms:
            z = randoms['z']
            cosmo = self.data_block[section_names.fiducial_cosmology,'cosmo']
        else:
            z = randoms['distance']
            cosmo = None
        nbar = utils.RedshiftDensityInterpolator(redshifts,weights=randoms['weight_comp'],cosmo=cosmo,**nbar,**randoms.mpi_attrs)
        for name,catalog in catalogs.items():
            if 'z' in randoms:
                catalog['nbar'] = nbar(catalog['z'])
            else:
                catalog['nbar'] = nbar(catalog['distance'])
    else:
        from_origin('nbar')

    if weight_fkp is None:
        for name,catalog in catalogs.items():
            catalog['weight_fkp'] = 1./(1. + catalog['nbar']*P0_fkp)
    else:
        from_origin('weight_fkp')

    for name,catalog in catalogs.items():
        catalog['weight'] = catalog['weight_comp']*catalog['weight_fkp']

    return catalogs['data'],catalogs['randoms']


def prepare_box_catalog(data, position='Position', weight=None):

    origin_data = data
    data = origin_data.copy(columns=[])
    data['position'] = origin_data.eval(position)
    if weight is None:
        data['weight'] = data.ones()
    else:
        data['weight'] = origin_data.eval(weight)
    return data
