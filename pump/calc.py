import matplotlib.pyplot as plt
import numpy as np
import seawater as sw
import xarray as xr
import warnings

import dcpy

def calc_reduced_shear(data):
    '''
    Estimate reduced shear for a dataset. Dataset must contain 'u', 'v', 'depth', 'dens'.
    '''

    data['S2'] = (data.u.differentiate('depth')**2 +
                  data.v.differentiate('depth')**2)
    data['S2'].attrs['long_name'] = '$S^2$'
    data['S2'].attrs['units'] = 's$^{-2}$'

    data['shear'] = np.sqrt(data.S2)
    data['shear'].attrs['long_name'] = '|$u_z$|'
    data['shear'].attrs['units'] = 's$^{-1}$'

    data['N2'] = (9.81 * 1.7e-4 * data.theta.differentiate('depth')
                  - 9.81 * 7.6e-4 * data.salt.differentiate('depth'))
    data['N2'].attrs['long_name'] = '$N^2$'
    data['N2'].attrs['units'] = 's$^{-2}$'

    data['shred2'] = data.S2 - 4 * data.N2
    data.shred2.attrs['long_name'] = 'Reduced shear$^2$'
    data.shred2.attrs['units'] = '$s^{-2}$'

    data['Ri'] = data.N2 / data.S2
    data.Ri.attrs['long_name'] = 'Ri'
    data.Ri.attrs['units'] = ''

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


def get_dcl_base_shear(data):
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

    dcl_max = _get_max(s2.where(s2.depth < -20, 0)
                       .where(s2.depth > euc_max, 0), 'depth')

    dcl_max.attrs['long_name'] = 'DCL Base (shear)'
    dcl_max.attrs['units'] = 'm'
    dcl_max.attrs['description'] = 'Depth of maximum total shear squared (above EUC)'

    return dcl_max


def get_dcl_base_Ri(data):
    '''
    Estimates base of the deep cycle layer as max depth where Ri <= 0.25.

    References
    ----------

    Lien et. al. (1995)
    Pham et al (2017)
    '''

    if 'Ri' not in data:
        raise ValueError('Ri not in provided dataset.')

    if 'euc_max' not in data:
        euc_max = get_euc_max(data.u)
    else:
        euc_max = data.euc_max


    depth = xr.broadcast(data.Ri, data.depth)[1]

    dcl_max = depth.where((data.Ri < 0.25)).min('depth')

    dcl_max.attrs['long_name'] = 'DCL Base (Ri)'
    dcl_max.attrs['units'] = 'm'
    dcl_max.attrs['description'] = 'Deepest depth above EUC where Ri=0.25'

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

    thresh = xr.where((drho > 0.01) & (N2 > 1e-5), drho.depth, np.nan)
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
    labels = []
    peak_kwargs = {'prominence': 0.02}

    for dd in v[dim2]:
        vsub = v.sel({dim2: dd})
        if debug:
            f, ax = plt.subplots(2, 1, sharex=True, constrained_layout=True)

            vsub.plot(ax=ax[0], x='time')
            dcpy.plots.liney(0, ax=ax[0])

        zeros = (zeros_da.sel({dim2: dd})
                 .dropna('time').values.astype(np.int32))

        zeros_unique = zeros[np.insert(np.diff(zeros), 0, 100) > 1]

        phase_0 = sp.signal.find_peaks(vsub, **peak_kwargs)
        phase_90 = [zeros_unique[np.nonzero(dvdt.sel({dim2: dd}).values[zeros_unique] < 0)[0]]]
        phase_180 = sp.signal.find_peaks(-vsub, **peak_kwargs)
        phase_270 = [zeros_unique[np.nonzero(dvdt.sel({dim2: dd}).values[zeros_unique] > 0)[0]]]

        vamp = np.abs(vsub
                      .isel(time=np.sort(np.hstack([phase_0[0], phase_90[0],
                                                    phase_180[0], phase_270[0]])))
                      .diff('time', label='lower'))

        # One version with phase=0 at points in phase_0
        # One version with 360 at points in phase_0
        # Then merge sensibly
        phase = xr.zeros_like(vsub) * np.nan
        label = xr.zeros_like(vsub) * np.nan
        phase2 = phase.copy(deep=True)
        start_num = 1
        for pp, cc, ph in zip([phase_0, phase_90, phase_180, phase_270],
                              'rgbk',
                              [0, 90, 180, 270]):
            idx = pp[0]

            # 0 phase must be positive v
            if ph == 0:
                if not np.all(vsub[idx] > 0):
                    idx = np.where(vsub[idx] > 0, idx, np.nan)
                    idx = idx[~np.isnan(idx)].astype(np.int32)

                label.values[idx] = np.arange(start_num, len(idx)+1)

            # 180 phase must be negative v
            if ph == 180:
                if not np.all(vsub[idx] < 0):
                    idx = np.where(vsub[idx] < 0, idx, np.nan)
                    idx = idx[~np.isnan(idx)].astype(np.int32)

            if debug:
                vsub[idx].plot(ax=ax[0], color=cc, ls='none', marker='o')
                ax[1].plot(vsub.time[idx], ph*np.ones_like(idx), color=cc, ls='none', marker='o')

            phase[idx] = ph
            if ph < 10:
                phase2[idx] = 360
            else:
                phase2[idx] = ph

        if not (np.all(np.isin(phase.dropna('time').diff('time'), [90, -270]))
                or np.all(np.isin(phase2.dropna('time').diff('time'), [90, -270]))):
            warnings.warn('Secondary peaks detected!')

        phase = phase.interpolate_na('time', method='linear')
        phase2 = phase2.interpolate_na('time', method='linear')

        dpdt = phase.differentiate('time')

        phase_new = xr.where((phase2 >= 270) & (phase2 < 360)
                             & (phase < 270) & (dpdt <= 0),
                             phase2, phase)
        vampf = vamp.reindex(time=phase.time).ffill('time')
        phase_new = phase_new.where(vampf > 0.1)

        label = label.ffill('time')

        # periods don't necessarily start with phase = 0
        phase_no_period = np.logical_and(~np.isnan(phase_new), np.isnan(label))
        label.values[phase_no_period.values] = 0

        if np.any(label == 0):
           label += 1

        if debug:
            # vampf.plot.step(ax=ax[0])
            phase_new.plot(ax=ax[1])
            dcpy.plots.liney([0, 90, 180, 270, 360], ax=ax[1])
            ax2 = ax[1].twinx()
            (label.ffill('time')
             .plot(ax=ax2, x='time', color='k'))
            ax[0].set_xlabel('')
            ax[1].set_title('');
            ax[1].set_ylabel('TIW phase [deg]')

        for dd in set(list(phase_new.coords))-set(['time']):
            phase_new = phase_new.expand_dims(dd)
            label = label.expand_dims(dd)

        phases.append(phase_new)
        labels.append(label)

    phase = xr.merge(phases).to_array().squeeze()
    phase.attrs['long_name'] = 'TIW phase'
    phase.name = 'tiw_phase'
    phase.attrs['units'] = 'deg'

    label = xr.merge(labels).to_array().squeeze().ffill('time')
    label.name = 'period'

    phase = xr.merge([phase, label])

    if unstack:
        # lost stack information earlier; re-assign that
        phase['stacked'] = v['stacked']
        phase = phase.unstack('stacked')

    # phase['period'] = phase.period.where(~np.isnan(phase.tiw_phase))

    # get rid of 1 point periods
    mask = phase.tiw_phase.groupby(phase.period).count() == 1
    drop_num = mask.period.where(mask, drop=True).values
    phase['period'] = (phase['period']
                       .where(np.logical_not(phase.period.isin(drop_num))))

    return phase
