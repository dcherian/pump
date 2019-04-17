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
