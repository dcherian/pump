import numpy as np
import time
import xarray as xr

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

        self.latitude = self.surface.latitude
        self.longitude = self.surface.longitude
        self.time = self.surface.time

        self.mid_year = np.unique(self.time.dt.year)[1]

        self.domain = dict()
        self.domain['xyt'] = dict()
        for dim in ['latitude', 'longitude', 'time']:
            self.domain['xyt'][dim] = slice(
                self.surface[dim].values.min(), self.surface[dim].values.max())
        self.domain['xy'] = {'latitude': self.domain['xyt']['latitude'],
                             'longitude': self.domain['xyt']['longitude']}

        try:
            self.mean = (xr.open_dataset(self.dirname + '/obs_subset/annual-mean.nc')
                         .squeeze())
        except FileNotFoundError:
            self.mean = None

        class obs_container:
            pass

        self.obs = obs_container()

        try:
            self.tao = xr.open_dataset(self.dirname + '/obs_subset/tao-extract.nc')
        except FileNotFoundError:
            self.tao = None

        try:
            self.johnson = xr.open_dataset(dirname + '/obs_subset/johnson-section-mean.nc')
        except FileNotFoundError:
            self.johnson = None

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
