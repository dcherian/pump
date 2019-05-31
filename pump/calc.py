import matplotlib.pyplot as plt
import numpy as np
import seawater as sw
import xarray as xr

import dcpy

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

    v.attrs['long_name'] = 'v: (10, 80m) avg, 10d lowpass'

    if v.ndim == 1:
        v = v.expand_dims('new_dim')
        unstack = False
    elif v.ndim > 2:
        unstack = True
        v = v.stack({'stacked': set(v.dims) - set(['time'])})
    else:
        unstack = False

    dvdt = v.differentiate('time')

    zeros_da = xr.where(np.abs(v) < 1e-2,
                        xr.DataArray(np.arange(v.shape[v.get_axis_num('time')]),
                                     dims=['time'],
                                     coords={'time': v.time}),
                        np.nan)

    assert v.ndim == 2
    dim2 = list(set(v.dims) - set(['time']))[0]

    phases = []
    peak_kwargs = {'prominence': 0.05}

    for dd in v[dim2]:
        if debug:
            f, ax = plt.subplots(2, 1, sharex=True, constrained_layout=True)

            v.sel({dim2: dd}).plot(ax=ax[0], x='time')
            dcpy.plots.liney(0, ax=ax[0])

        zeros = (zeros_da.sel({dim2: dd})
                 .dropna('time').values.astype(np.int32))

        zeros_unique = zeros[np.insert(np.diff(zeros), 0, 100) > 1]

        phase_0 = sp.signal.find_peaks(v.sel({dim2: dd}), **peak_kwargs)
        phase_90 = [zeros_unique[np.nonzero(dvdt.sel({dim2: dd}).values[zeros_unique] < 0)[0]]]
        phase_180 = sp.signal.find_peaks(-v.sel({dim2: dd}), **peak_kwargs)
        phase_270 = [zeros_unique[np.nonzero(dvdt.sel({dim2: dd}).values[zeros_unique] > 0)[0]]]

        # One version with phase=0 at points in phase_0
        # One version with 360 at points in phase_0
        # Then merge sensibly
        phase = xr.zeros_like(v.sel({dim2: dd})) * np.nan
        phase2 = phase.copy(deep=True)
        for pp, cc, ph in zip([phase_0, phase_90, phase_180, phase_270],
                              'rgbk',
                              [0, 90, 180, 270]):
            if debug:
                v.sel({dim2: dd})[pp[0]].plot(ax=ax[0], color=cc, ls='none', marker='o')

            phase[pp[0]] = ph
            if ph < 10:
                phase2[pp[0]] = 360
            else:
                phase2[pp[0]] = ph

        if not (np.all(np.isin(phase.dropna('time').diff('time'), [90, -270]))
                or np.all(np.isin(phase2.dropna('time').diff('time'), [90, -270]))):
            raise AssertionError('Secondary peaks detected!')

        phase = phase.interpolate_na('time', method='linear')
        phase2 = phase2.interpolate_na('time', method='linear')

        dpdt = phase.differentiate('time')

        phase_new = xr.where((phase2 >= 270) & (phase2 < 360)
                             & (phase < 270) & (dpdt <= 0),
                             phase2, phase)
        if debug:
            phase_new.plot(ax=ax[1])
            dcpy.plots.liney([0, 90, 180, 270, 360], ax=ax[1])
            ax[0].set_xlabel('')
            ax[1].set_title('');
            ax[1].set_ylabel('TIW phase [deg]')

        for dd in set(list(phase_new.coords))-set(['time']):
            phase_new = phase_new.expand_dims(dd)

        phases.append(phase_new)

    phase = xr.merge(phases)

    if unstack:
        # lost stack information earlier; re-assign that
        phase['stacked'] = v['stacked']
        phase = phase.unstack('stacked')

    phase = phase.to_array().squeeze()

    phase.attrs['long_name'] = 'TIW phase'
    phase.name = 'tiw_phase'
    phase.attrs['units'] = 'deg'

    return phase
