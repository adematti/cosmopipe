from cosmopipe.lib.catalog import utils


def prepare_survey_angular_catalogs(data, randoms=None, ra='RA', dec='DEC', weight_comp=None):

    origin_catalogs = {'data':data}
    if randoms is not None: origin_catalogs['randoms'] = randoms
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

    return catalogs['data'], catalogs.get('randoms',None)


def prepare_survey_catalogs(data, randoms=None, cosmo=None, ra='RA', dec='DEC', z='Z', position=None, weight_comp=None, nbar='NZ', weight_fkp=None, P0_fkp=0.,\
                            zmin=0,zmax=10.,ramin=-10.,ramax=400.):

    catalogs = {}
    origin_catalogs = {'data':data}
    if randoms is not None: origin_catalogs['randoms'] = randoms
    catalogs = {name:catalog.copy(columns=[]) for name,catalog in origin_catalogs.items()}

    def from_origin(origin_column, column):
        for name,catalog in catalogs.items():
            #just need to get this done - figure out ideal later
            if origin_column=='WEIGHTBOSS':
                if name=='data':
                    catalog[column] = origin_catalogs[name]['WEIGHT_SYSTOT']* \
                    (origin_catalogs[name]['WEIGHT_CP']+ \
                     origin_catalogs[name]['WEIGHT_NOZ'] - 1)
                else: catalog[column] = origin_catalogs[name].ones()
            else: catalog[column] = origin_catalogs[name].eval(origin_column)
    if weight_comp is None:
        for name,catalog in catalogs.items():
            catalog['weight_comp'] = origin_catalogs[name].ones()
    else:
        from_origin(weight_comp,'weight_comp')
    #read even with positions in case want to cut on these
    from_origin(z,'z')
    from_origin(ra,'ra')
    if position is None:
        from_origin(dec,'dec')
        for name,catalog in catalogs.items():
            catalog['distance'] = cosmo.get_background().comoving_radial_distance(catalog['z'])
            catalog['position'] = utils.sky_to_cartesian(catalog['distance'],catalog['ra'],catalog['dec'],degree=True)
    else:
        from_origin(position,'position')
        for name,catalog in catalogs.items():
            catalog['distance'] = utils.distance(catalog['position'])

    if isinstance(nbar,dict):
        randoms = catalogs.get('randoms',None)
        #added because incoming data doesn't have weight_comp
        data = catalogs['data']
        use_randoms = randoms is not None
        if use_randoms:
            alpha = data.sum('weight_comp')/randoms.sum('weight_comp')
        else:
            randoms = catalogs['data']
            alpha = 1.
        if 'z' in randoms:
            z = randoms['z']
            radial_distance = cosmo.get_background().comoving_radial_distance
        else:
            z = randoms['distance']
            radial_distance = None
        #had redshifts here which doesn't exist
        #running into AttributeError: 'Catalog' object has no attribute 'mpi_attrs'
        nbar = utils.RedshiftDensityInterpolator(randoms['z'],weights=randoms['weight_comp'],radial_distance=radial_distance,**nbar,**randoms.mpi_attrs)
        for name,catalog in catalogs.items():
            if 'z' in randoms:
                catalog['nbar'] = alpha*nbar(catalog['z'])
            else:
                catalog['nbar'] = alpha*nbar(catalog['distance'])
    elif isinstance(nbar,float):
        for name,catalog in catalogs.items():
            catalog['nbar'] = catalog.ones()*nbar
    else:
        from_origin(nbar,'nbar')

    if weight_fkp is None:
        for name,catalog in catalogs.items():
            catalog['weight_fkp'] = 1./(1. + catalog['nbar']*P0_fkp)
    else:
        from_origin(weight_fkp,'weight_fkp')

    for name,catalog in catalogs.items():
        catalog['weight'] = catalog['weight_comp']*catalog['weight_fkp']

    for name,catalog in catalogs.items():
        mask=(catalog['z']>=zmin) & (catalog['z']<zmax) & (catalog['ra']<ramax) & (catalog['ra']>=ramin)
        for ind in catalog.columns():
          catalog[ind]=catalog[ind][mask]
    return catalogs['data'],catalogs.get('randoms',None)


def prepare_box_catalog(data, position='Position', weight=None):

    origin_data = data
    data = origin_data.copy(columns=[])
    data['position'] = origin_data.eval(position)
    if weight is None:
        data['weight'] = data.ones()
    else:
        data['weight'] = origin_data.eval(weight)
    return data
