from cosmopipe.lib.catalog import utils


def prepare_survey_angular_catalogs(data, randoms, ra='RA', dec='DEC', weight_comp=None):

    origin_catalogs = {'data':data,'randoms':randoms}
    catalogs = {name:catalog.copy(columns=[]) for name,catalog in origin_catalogs.items()}

    def from_origin(origin_column, column):
        for name,catalog in catalogs.items():
            catalog[column] = origin_catalogs[name].eval(origin_column)

    if weight_comp is None:
        for name,catalog in catalogs.items():
            catalog['weight_comp'] = origin_catalogs[name].ones()
    else:
        from_origin(weight_comp,'weight_comp')

    from_origin(ra,'ra')
    from_origin(dec,'dec')

    return catalogs['data'],catalogs['randoms']


def prepare_survey_catalogs(data, randoms, cosmo=None, ra='RA', dec='DEC', z='Z', position=None, weight_comp=None, nbar='NZ', weight_fkp=None, P0_fkp=0.):

    origin_catalogs = {'data':data,'randoms':randoms}

    catalogs = {}
    catalogs['data'],catalogs['randoms'] = prepare_survey_angular_catalogs(data,randoms,ra=ra,dec=dec,weight_comp=weight_comp)

    def from_origin(origin_column, column):
        for name,catalog in catalogs.items():
            catalog[column] = origin_catalogs[name].eval(origin_column)

    if z is not None:
        from_origin(z,'z')

    if position is None:
        for name,catalog in catalogs.items():
            catalog['distance'] = cosmo.get_background().comoving_radial_distance(catalog['z'])
            catalog['position'] = utils.sky_to_cartesian(catalog['distance'],catalog['ra'],catalog['dec'],degree=True)
    else:
        from_origin(position,'position')
        for name,catalog in catalogs.items():
            catalog['distance'] = utils.distance(catalog['position'])

    if isinstance(nbar,dict):
        if 'z' in randoms:
            z = randoms['z']
            radial_distance = cosmo.get_background().comoving_radial_distance
        else:
            z = randoms['distance']
            radial_distance = None
        nbar = utils.RedshiftDensityInterpolator(redshifts,weights=randoms['weight_comp'],radial_distance=radial_distance,**nbar,**randoms.mpi_attrs)
        for name,catalog in catalogs.items():
            if 'z' in randoms:
                catalog['nbar'] = nbar(catalog['z'])
            else:
                catalog['nbar'] = nbar(catalog['distance'])
    else:
        from_origin(nbar,'nbar')

    if weight_fkp is None:
        for name,catalog in catalogs.items():
            catalog['weight_fkp'] = 1./(1. + catalog['nbar']*P0_fkp)
    else:
        from_origin(weight_fkp,'weight_fkp')

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
