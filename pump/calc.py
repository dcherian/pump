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
    '''
    Estimates base of the deep cycle layer as the depth of max total shear squared.

    References
    ----------

    Inoue et al. (2012)
    '''

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


def get_mld(dens):
    '''
    Given density field, estimate MLD as depth where Δρ > 0.015 and N2 > 2e-5.
    Interpolates density to 1m gridl
    '''

    drho = dens.interp(depth=np.arange(0, -200, -1)) - dens.isel(depth=0)
    N2 = -9.81/1025 * dens.interp(depth=np.arange(0, -200, -1)).differentiate('depth')

    thresh = xr.where((drho > 0.015) & (N2 > 2e-5), drho.depth, np.nan)
    mld = thresh.max('depth')

    return mld


def get_tiw_phase(v, debug=False):
    '''
    Estimates TIW phase using 10 day low-passed meridional velocity
    averaged between 10m and 80m.

    Input
    -----
    v: xr.DataArray
        Meridional velocity (z, t).

    Output
    ------
    phase: xr.DataArray
        Phase in degrees.

    References
    ----------
    Inoue et. al. (2019)
    '''

    import scipy as sp
    import xfilter

    v = xfilter.lowpass(
        v.sel(depth=slice(-10, -80)).mean('depth'),
        coord='time',
        freq=1/10.0,
        cycles_per='D')

    dvdt = v.differentiate('time')

    phase = xr.zeros_like(v) * np.nan

    zeros = (xr.where(np.abs(v) < 1e-2, np.arange(v.size), np.nan)
             .dropna('time').values.astype(np.int32))
    zeros_unique = zeros[np.insert(np.diff(zeros), 0, 100) > 1]

    phase_0 = sp.signal.find_peaks(v)
    phase_90 = [zeros_unique[np.nonzero(dvdt.values[zeros_unique] < 0)[0]]]
    phase_180 = sp.signal.find_peaks(-v)
    phase_270 = [zeros_unique[np.nonzero(dvdt.values[zeros_unique] > 0)[0]]]

    # One version with phase=0 at points in phase_0
    # One version with 360 at points in phase_0
    # Then merge sensibly
    phase2 = phase.copy(deep=True)
    for pp, cc, ph in zip([phase_0, phase_90, phase_180, phase_270],
                          'rgbk',
                          [0, 90, 180, 270]):
        if debug:
            v[pp[0]].plot(color=cc, ls='none', marker='o')
        phase[pp[0]] = ph
        if ph < 10:
            phase2[pp[0]] = 360
        else:
            phase2[pp[0]] = ph

    phase = phase.interpolate_na('time', method='linear')
    phase2 = phase2.interpolate_na('time', method='linear')

    if debug:
        v.plot(x='time')
        dcpy.plots.liney(0)

        plt.figure()
        phase.plot()
        phase2.plot()

    dpdt = phase.differentiate('time')

    phase_new = xr.where((phase2 >= 270) & (phase2 < 360)
                         & (phase < 270) & (dpdt <= 0),
                         phase2, phase)
    if debug:
        phase_new.plot()

    phase_new.attrs['long_name'] = 'TIW phase'
    phase_new.name = 'tiw_phase'
    phase_new.attrs['units'] = 'deg'

    return phase_new
