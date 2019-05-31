import numpy as np
import time
import xarray as xr
import xmitgcm

from ..calc import (calc_reduced_shear, get_euc_max, get_dcl_base, get_mld,
                    get_tiw_phase)
from ..constants import *
from ..obs import *

class model:

    from . import validate

    def __init__(self, dirname, name, kind='mitgcm', full=False, budget=False):
        self.dirname = dirname
        self.kind = kind
        self.name = name

        try:
            self.surface = (xr.open_dataset(self.dirname + '/obs_subset/surface.nc')
                            .squeeze())
        except FileNotFoundError:
            self.surface = xr.Dataset()

        # self.surface['theta_anom'] = self.surface.theta - self.surface.theta.mean(['longitude', 'time'])

        if full:
            self.read_full()
        else:
            self.full = xr.Dataset()
            self.depth = None

        if budget:
            self.read_budget()
        else:
            self.budget = xr.Dataset()

        self.domain = dict()
        self.domain['xyt'] = dict()

        if self.domain['xyt']:
            self.oisst = read_sst(self.domain['xyt'])

        self.update_coords()
        self.read_metrics()

        try:
            self.mean = (xr.open_dataset(self.dirname + '/obs_subset/annual-mean.nc')
                         .squeeze())
        except FileNotFoundError:
            self.mean = None

        class obs_container:
            pass

        self.obs = obs_container()

        self.read_tao()

        try:
            self.johnson = xr.open_dataset(dirname + '/obs_subset/johnson-section-mean.nc')
        except FileNotFoundError:
            self.johnson = None

        self.tiw_trange = [slice('1995-10-01', '1996-03-01'),
                           slice('1996-09-01', '1997-03-01')]

    def __repr__(self):
        string = f'{self.name} [{self.dirname}]'
        # Add resolution

        return string

    def extract_johnson_sections(self):
        (self.full.sel(longitude=section_lons, method='nearest')
         .sel(time=str(self.mid_year))
         .mean('time')
         .load()
         .to_netcdf(self.dirname + '/obs_subset/johnson-section-mean.nc'))

    def extract_tao(self):
        self.tao = (
            self.full.sel(
                longitude=[-170, -155, -140, -125, -110, -95],
                latitude=[-8, -5, -2, 0, 2, 5, 8],
                method='nearest')
            .sel(depth=slice(0, -500)))

        # round lat, lon
        self.tao['latitude'].values = np.array([-8, -5, -2, 0, 2, 5, 8]) * 1.0
        self.tao['longitude'].values = (
            np.array([-170, -155, -140, -125, -110, -95]) * 1.0)

        (self.tao.load()
         .to_netcdf(self.dirname + '/obs_subset/tao-extract.nc'))

    def read_full(self):
        start_time = time.time()

        if self.kind == 'mitgcm':
            self.full = (xr.open_mfdataset(self.dirname + '/Day_[0-9][0-9][0-9][0-9].nc',
                                           engine='h5netcdf', parallel=True))

        if self.kind == 'roms':
            self.full = xr.Dataset()

        print('Reading all files took {time} seconds'.format(
                time=time.time()-start_time))

        self.depth = self.full.depth

    def read_metrics(self):
        dirname = self.dirname + '../'

        h = dict()
        for ff in ['hFacC', 'RAC', 'RF']:
            try:
                h[ff] = xmitgcm.utils.read_mds(dirname + ff, dask_delayed=False)[ff]
            except (FileNotFoundError, OSError):
                print('metrics files not available.')
                return xr.Dataset()

        hFacC = h['hFacC'].copy().squeeze().astype('float32')
        RAC = h['RAC'].copy().squeeze().astype('float32')
        RF = h['RF'].copy().squeeze().astype('float32')

        del h

        RAC = xr.DataArray(RAC,
                           dims=['latitude', 'longitude'],
                           coords={'longitude': self.longitude,
                                   'latitude': self.latitude},
                           name='RAC')

        self.depth = xr.DataArray((RF[1:] + RF[:-1])/2, dims=['depth'],
                                  name='depth',
                                  attrs={'long_name': 'depth',
                                         'units': 'm'})

        dRF = xr.DataArray(np.diff(RF.squeeze()),
                           dims=['depth'],
                           coords={'depth': self.depth},
                           name='dRF',
                           attrs={'long_name': 'cell_height',
                                  'units': 'm'})

        RF = xr.DataArray(RF.squeeze(),
                          dims=['depth_left'],
                          name='depth_left')

        hFacC = xr.DataArray(hFacC, dims=['depth', 'latitude', 'longitude'],
                             coords={'depth': self.depth,
                                     'latitude': self.latitude,
                                     'longitude': self.longitude},
                             name='hFacC')

        metrics = xr.merge([dRF, hFacC, RAC])

        metrics['cellvol'] = np.abs(metrics.RAC * metrics.dRF * metrics.hFacC)

        metrics['cellvol'] = metrics.cellvol.where(metrics.cellvol > 0)

        metrics['RF'] = RF

        self.metrics = metrics

    def read_budget(self):

        kwargs = dict(engine='h5netcdf',
                      parallel=True,
                     )

        files = sorted(glob.glob(self.dirname + 'Day_*_hb.nc'))
        self.budget = xr.merge([
            xr.open_mfdataset(files,
                              drop_variables=['DFxE_TH', 'DFyE_TH', 'DFrE_TH'],
                              **kwargs),
            xr.open_mfdataset(self.dirname + 'Day_*_sf.nc',
                              **kwargs)
        ])

        self.budget['oceQsw'] = self.budget.oceQsw.fillna(0)

    def get_tiw_phase(self, v, debug=False):

        ph = []
        for tt in self.tiw_trange:
            ph.append(get_tiw_phase(v.sel(time=tt), debug=debug))

        phase = xr.merge(ph).drop('variable').reindex(time=v.time)

        return phase['tiw_phase']

    def read_tao(self):
        try:
            self.tao = xr.open_mfdataset(self.dirname + '/obs_subset/tao-*.nc')
        except FileNotFoundError:
            self.tao = None
            return

        self.tao = calc_reduced_shear(self.tao)
        self.tao['euc_max'] = get_euc_max(self.tao.u)
        self.tao['dcl_base'] = get_dcl_base(self.tao)
        self.tao['mld'] = get_mld(self.tao.dens)

        CV = (self.metrics.cellvol
              .sel(latitude=self.tao.latitude,
                   longitude=self.tao.longitude,
                   depth=self.tao.depth,
                   method='nearest')
              .assign_coords(**dict(self.tao.isel(time=1).coords)))

        self.tao['Jq'] = (1035 * 3999 * 10 * self.tao.DFrI_TH / CV)
        self.tao['Jq'].attrs['long_name'] = "$J_q^t$"
        self.tao['Jq'].attrs['units'] = 'W/m$^2$'

    def update_coords(self):
        if self.surface:
            ds = self.surface
        elif self.full:
            ds = self.full
        elif self.budget:
            ds = self.budget

        self.latitude = ds.latitude
        self.longitude = ds.longitude
        self.time = ds.time
        self.mid_year = np.unique(self.time.dt.year)[1]

        if 'depth' in ds.variables and len(ds['depth']) > 1:
            self.depth = ds.depth

        for dim in ['latitude', 'longitude', 'time']:
            self.domain['xyt'][dim] = slice(
                getattr(self, dim).values.min(),
                getattr(self, dim).values.max())

        self.domain['xy'] = {'latitude': self.domain['xyt']['latitude'],
                             'longitude': self.domain['xyt']['longitude']}
