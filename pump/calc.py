import numpy as np
import seawater as sw
import xarray as xr

def calc_reduced_shear(data):
    '''
    Estimate reduced shear for a dataset. Dataset must contain 'u', 'v', 'depth', 'dens'.
    '''


    data['shear'] = np.hypot(data.u.differentiate('depth'), data.v.differentiate('depth'))
    data['dens'] = data.salt.copy(data=sw.pden(data.salt,
                                               data.theta,
                                               xr.broadcast(data.theta, data.depth)[1], 0))
    data['N2'] = -9.81/1025 * data.dens.differentiate('depth')

    data['shred2'] = data.shear**2 - 4 * data.N2
    data.shred2.attrs['long_name'] = 'Reduced shear$^2$'
    data.shred2.attrs['units'] = '$s^{-2}$'

    return data


def _get_max(var, dim='depth'):

    #return((xr.where(var == var.max(dim), var[dim], np.nan))
    #       .max(dim))

    coords = dict(var.coords)
    coords.pop(dim)

    dims = list(var.dims)
    del dims[var.get_axis_num(dim)]

    argmax = np.argmax(var.values, var.get_axis_num(dim))
    da = xr.DataArray(argmax.squeeze(),
                      dims=dims,
                      coords=coords)

    return var[dim][da].drop(dim)


def get_euc_max(u):
    ''' Given a u field, returns depth of max speed i.e. EUC maximum. '''

    euc_max = _get_max(u, 'depth')
    euc_max.attrs['long_name'] = 'Depth of EUC max'
    euc_max.attrs['units'] = 'm'

    return euc_max


def get_dcl_base(data):

    if 'shear' in data:
        s2 = data['shear']**2
    else:
        s2 = np.hypot(data.u.differentiate('depth'), data.v.differentiate('depth'))**2

    if 'euc_max' not in data:
        euc_max = get_euc_max(data.u)
    else:
        euc_max = data.euc_max

    dcl_max = _get_max(s2.where(s2.depth > euc_max, 0), 'depth')

    dcl_max.attrs['long_name'] = 'Depth of DCL Base'
    dcl_max.attrs['units'] = 'm'
    dcl_max.attrs['description'] = 'Depth of maximum total shear squared (above EUC)'

    return dcl_max


def get_euc_transport(u):
    euc = u.where(u > 0).sel(latitude=slice(-3, 3, None))
    euc.values[np.isnan(euc.values)] = 0
    euc = euc.integrate('latitude') * 0.1

    return (euc)


def calc_tao_ri(adcp, temp):
    '''

    Calculate Ri for TAO dataset.
    Interpolates to 5m grid and then differentiates.
    Uses N^2 = g alpha dT/dz


    Inputs
    ------

    adcp: xarray.Dataset
        Dataset with ['u', 'v']

    temp: xarry.DataArray
        Temperature DataArray

    References
    ----------

    Smyth & Moum (2013)
    Pham et al. (2017)

    '''

    V = adcp[['u', 'v']].load()
    V['shear'] = np.hypot(V['u'].differentiate('depth'),
                          V['v'].differentiate('depth'))

    T = (temp.load()
         .where(temp.depth > -500, drop=True)
         .dropna('depth', how='all')
         .sel(time=V.time)
         .sortby('depth')
         .interpolate_na('depth', 'linear')
         .sortby('depth', 'descending')
         .interp(depth=V.depth))

    # the calculation is sensitive to using sw.alpha! can't just do 1.7e-4
    N2 = (9.81
          * sw.alpha(35, T, xr.broadcast(T, T.depth)[1])
          * T.differentiate('depth'))
    S2 = V.shear ** 2

    N2 = N2
    Ri = ((N2.where(N2 > 1e-7) / S2.where(S2 > 1e-10))
          .dropna('depth', how='all'))

    return Ri
