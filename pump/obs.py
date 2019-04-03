import xarray as xr


def read_johnson(filename = '../glade/obs/johnson-eq-pac-adcp.cdf'):
    ds = (xr.open_dataset(filename)
            .rename({'XLON': 'longitude',
                    'YLAT11_101': 'latitude',
                    'ZDEP1_50': 'depth',
                    'POTEMPM': 'temp',
                    'SALINITYM': 'salt',
                    'SIGMAM': 'rho',
                    'UM': 'u',
                    'XLONedges': 'lon_edges'}))

    ds['longitude'] -= 360
    ds['depth'] *= -1
    ds['depth'].attrs['units'] = 'm'
    ds['u'].attrs['units'] = 'm/s'

    return ds

def read_tao():
    tao = (xr.open_mfdataset(['../glade/obs/tao/'+ff
                              for ff in ['t_xyzt_dy.cdf', 's_xyzt_dy.cdf', 'cur_xyzt_dy.cdf']],
                             parallel=True)
           .rename({'U_320': 'u',
                    'V_321': 'v',
                    'T_20': 'temp',
                    'S_41': 'salt',
                    'lon': 'longitude',
                    'lat': 'latitude'
                   }))
    tao['longitude'] -= 360

    tao['u'] /= 100
    tao['v'] /= 100

    tao = tao.sel(**domain).drop(['S_300', 'D_310'])

    for vv in tao:
        tao[vv] = tao[vv].where(tao[vv] < 1e4)
        tao[vv].attrs['long_name'] = ''

    return tao

def read_sst():
    sst = xr.open_mfdataset(['../glade/obs/oisst/sst.day.mean.'+str(yy)+'.nc' for yy in [1995, 1996, 1997]],
                            parallel=True)

    sst['lon'] -= 360

    sst = sst.rename({'lat': 'latitude', 'lon': 'longitude'}).sel(**mitgcm.domain['xyt'])

    sst['anom'] = sst.sst - sst.sst.mean(['longitude', 'time'])
    sst['anom'].attrs['long_name'] = 'OISST Anomaly'
    sst['anom'].attrs['units'] = r'$\degree$C'
    sst['anom'].attrs['description'] = 'SST - mean(SST) in longitude, time after subsetting to simulation time length'

    return sts
