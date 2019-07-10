import dcpy

import numpy as np
import pandas as pd
import xarray as xr

from .constants import *

root = '/glade/p/nsc/ncgd0043/'


def read_all(domain=None):
    johnson = read_johnson()
    tao = read_tao(domain)
    sst = read_sst(domain)
    oscar = read_oscar(domain)

    return [johnson, tao, sst, oscar]


def read_johnson(filename=root+'/obs/johnson-eq-pac-adcp.cdf'):
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


def read_tao_adcp(domain=None):
    adcp = (xr.open_dataset(root+'/obs/tao/adcp_xyzt_dy.cdf')
            .rename({'lon': 'longitude',
                     'lat': 'latitude',
                     'U_1205': 'u',
                     'V_1206': 'v'})
            .chunk({'latitude': 1, 'longitude': 1}))

    for vv in adcp:
        adcp[vv] /= 100
        adcp[vv].attrs['long_name'] = vv
        adcp[vv].attrs['units'] = 'm/s'
        adcp[vv] = adcp[vv].where(np.abs(adcp[vv]) < 1000)

    adcp['longitude'] -= 360
    adcp['depth'] *= -1
    adcp['depth'].attrs['units'] = 'm'
    adcp['u'].attrs['units'] = 'm/s'
    adcp['v'].attrs['units'] = 'm/s'

    if domain is not None:
        adcp = adcp.sel(**domain)
    else:
        adcp = adcp.sel(latitude=0)

    return (adcp
            .dropna('longitude', how='all')
            .dropna('depth', how='all'))

def read_tao(domain=None):
    tao = (xr.open_mfdataset([root+'/obs/tao/'+ff
                              for ff in ['t_xyzt_dy.cdf',
                                         's_xyzt_dy.cdf',
                                         'cur_xyzt_dy.cdf']],
                             parallel=False,
                             chunks={'lat': 1, 'lon': 1, 'depth': 5})
           .rename({'U_320': 'u',
                    'V_321': 'v',
                    'T_20': 'temp',
                    'S_41': 'salt',
                    'lon': 'longitude',
                    'lat': 'latitude'}))
    tao['longitude'] -= 360

    tao['u'] /= 100
    tao['v'] /= 100
    tao['u'].attrs['units'] = 'm/s'
    tao['v'].attrs['units'] = 'm/s'

    tao['depth'] *= -1

    tao = tao.drop(['S_300', 'D_310', 'QS_5300', 'QD_5310',
                    'QS_5041', 'SS_6041',
                    'QT_5020', 'ST_6020', 'SRC_6300'])

    for vv in tao:
        tao[vv] = tao[vv].where(tao[vv] < 1e4)
        tao[vv].attrs['long_name'] = ''

    if domain is not None:
        return tao.sel(**domain)
    else:
        return tao


def read_sst(domain=None):

    if domain is not None:
        years = range(pd.Timestamp(domain['time'].start).year,
                      pd.Timestamp(domain['time'].stop).year)
        sst = xr.open_mfdataset(
            [root+'/obs/oisst/sst.day.mean.'+str(yy)+'.nc' for yy in years],
            parallel=True)
    else:
        sst = xr.open_mfdataset(root+'/obs/oisst/sst.day.mean.*.nc',
                                parallel=True)

    sst['lon'] -= 360

    sst = sst.rename({'lat': 'latitude', 'lon': 'longitude'})

    sst['anom'] = sst.sst - sst.sst.mean(['longitude', 'time'])
    sst['anom'].attrs['long_name'] = 'OISST Anomaly'
    sst['anom'].attrs['units'] = r'$\degree$C'
    sst['anom'].attrs['description'] = ('SST - mean(SST) in longitude, '
                                        'time after subsetting to simulation '
                                        'time length')

    if domain is not None:
        return sst.sel(**domain)
    else:
        return sst


def read_oscar(domain=None):

    oscar = (dcpy.oceans.read_oscar(root+'/obs/oscar/')
             .rename({'lat': 'latitude', 'lon': 'longitude'}))
    oscar['longitude'] = oscar['longitude'] - 360
    oscar = oscar.sortby('latitude')

    if domain is not None:
        return oscar.sel(**domain)
    else:
        return oscar


def read_argo():

    dirname = root + '/obs/argo/'
    chunks = {'LATITUDE': 1, 'LONGITUDE': 1}

    argoT = xr.open_dataset(dirname + 'RG_ArgoClim_Temperature_2017.nc',
                            decode_times=False, chunks=chunks)
    argoS = xr.open_dataset(dirname + 'RG_ArgoClim_Salinity_2017.nc',
                            decode_times=False, chunks=chunks)

    argoS['S'] = argoS.ARGO_SALINITY_ANOMALY + argoS.ARGO_SALINITY_MEAN
    argoT['T'] = argoT.ARGO_TEMPERATURE_ANOMALY + argoT.ARGO_TEMPERATURE_MEAN

    argo = argoT.update(argoS)

    argo = (argo.rename({'LATITUDE': 'latitude',
                         'LONGITUDE': 'longitude',
                         'PRESSURE': 'depth',
                         'TIME': 'time',
                         'ARGO_TEMPERATURE_MEAN': 'Tmean',
                         'ARGO_TEMPERATURE_ANOMALY': 'Tanom',
                         'ARGO_SALINITY_MEAN': 'Smean',
                         'ARGO_SALINITY_ANOMALY': 'Sanom'}))

    _, ref_date = xr.coding.times._unpack_netcdf_time_units(
        argo.time.attrs['units'])

    argo.time.values = (pd.Timestamp(ref_date)
                        + pd.to_timedelta(30 * argo.time.values, unit='D'))
    argo['longitude'] -= 360
    argo['depth'] *= -1

    return argo


def process_nino34():
    nino34 = process_esrl_index('nina34.data')

    nino34.to_netcdf(root + '/obs/nino34.nc')


def process_oni():
    oni = process_esrl_index('oni.data', skipfooter=8)
    oni.to_netcdf(root + '/obs/oni.nc')


def process_esrl_index(file, skipfooter=3):
    ''' Read and make xarray version of climate indices from ESRL.'''

    month_names = (pd.date_range('01-Jan-2001', '31-Dec-2001', freq='MS')
                   .to_series()
                   .dt.strftime('%b').values
                   .astype(str))

    index = pd.read_csv(root+'/obs/'+file, index_col=0,
                        names=month_names, delim_whitespace=True,
                        skiprows=1, na_filter=False, skipfooter=skipfooter,
                        dtype=np.float32)

    flat = index.stack().reset_index()
    flat['time'] = (pd.date_range('01-jan-'+str(flat['level_0'].iloc[0]),
                                  '01-Jan-'+str(flat['level_0'].iloc[-1]+1),
                                  freq='M'))
    da = (flat.drop(['level_0', 'level_1'], axis=1)
          .rename({0: 'index'}, axis=1)
          .set_index('time')
          .to_xarray())

    return da.where(da > -90)['index']
