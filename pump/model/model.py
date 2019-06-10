import dcpy.plots
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seawater as sw
import time
import xarray as xr
import xmitgcm

from ..calc import (calc_reduced_shear, get_euc_max, get_dcl_base, get_mld,
                    get_tiw_phase)
from ..constants import *
from ..obs import *
from ..plot import plot_depths

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
        region = dict(longitude=[-170, -155, -140, -125, -110, -95],
                      latitude=[-8, -5, -2, 0, 2, 5, 8],
                      method='nearest')
        datasets = [self.full.sel(**region)
                    .sel(depth=slice(0, -500))
                    .load()]
        if self.budget:
            print('Merging in budget terms...')
            datasets.append(self.budget.sel(**region)
                            .sel(depth=slice(0, -500))
                            .load())
            if not self.full.time.equals(self.budget.time):
                datasets[0] = datasets[0].reindex(time=self.budget.time)

        self.tao = xr.merge(datasets)

        # round lat, lon
        self.tao['latitude'].values = np.array([-8, -5, -2, 0, 2, 5, 8]) * 1.0
        self.tao['longitude'].values = (
            np.array([-170, -155, -140, -125, -110, -95]) * 1.0)

        print('Writing to file...')
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
            if len(ph) > 1:
                start_num = ph[-2].period.max()
            else:
                start_num = 0
            ph[-1]['period'] += start_num

        phase = xr.merge(ph).drop('variable').reindex(time=v.time)

        return phase.set_coords('period')['tiw_phase']

    def read_tao(self):
        try:
            self.tao = xr.open_mfdataset(self.dirname + '/obs_subset/tao-*extract.nc')
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

        dz = np.abs(self.metrics.dRF[0])

        self.tao['Jq'] = (1035 * 3999 * dz * self.tao.DFrI_TH / CV)
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

        if 'depth' in ds.variables and not np.isscalar(ds['depth']):
            self.depth = ds.depth

        for dim in ['latitude', 'longitude', 'time']:
            self.domain['xyt'][dim] = slice(
                getattr(self, dim).values.min(),
                getattr(self, dim).values.max())

        self.domain['xy'] = {'latitude': self.domain['xyt']['latitude'],
                             'longitude': self.domain['xyt']['longitude']}

    def plot_tiw_composite(self, region=dict(latitude=0, longitude=-140),
                           ax=None, ds='tao', **kwargs):

        ds = getattr(self, ds)

        subset = ds.sel(**region)

        tiw_phase = self.get_tiw_phase(subset.v)
        subset = (subset.rename({'KPP_diffusivity': 'KT'})
                  .where(subset.depth < subset.mld - 5))

        for vv in ['mld', 'dcl_base', 'euc_max']:
            subset[vv] = subset[vv].max('depth')

        phase_bins = np.arange(0, 365, 10)
        grouped = subset.groupby_bins(tiw_phase, bins=phase_bins)
        mean = grouped.mean('time')

        mean['shear'] = mean['shear'] ** 2
        mean['shear'].attrs['long_name'] = '$S^2$'
        mean['shear'].attrs['units'] = 's$^{-2}$'

        if ax is None:
            f, axx = plt.subplots(5, 1, sharex=True, sharey=True,
                                  constrained_layout=True,
                                  gridspec_kw={'height_ratios': [1, 1, 1, 1, 1]})

            ax = dict(zip(['u', 'v', 'shear', 'N2', 'Jq', 'KT'], axx))
            f.set_size_inches((5, 6))

        else:
            axx = list(ax.values())

        cmaps = dict(u=mpl.cm.RdBu_r,
                     v=mpl.cm.RdBu_r,
                     shear=mpl.cm.Reds,
                     N2=mpl.cm.Blues,
                     Jq=mpl.cm.BuGn_r,
                     KT=mpl.cm.Reds)

        handles = dict()
        for aa in ax:
            if aa == 'KT':
                pkwargs = dict(norm=mpl.colors.LogNorm())
            elif aa == 'shear':
                pkwargs=dict(vmin=0, vmax=3e-4)
            elif aa == 'Jq':
                pkwargs=dict(vmax=0, vmin=-300)
            elif aa == 'u':
                pkwargs = dict(vmin=-0.8, vmax=0.8)
            elif aa == 'v':
                pkwargs = dict(vmin=-0.3, vmax=0.3)
            elif aa == 'N2':
                pkwargs = dict(vmin=0, vmax=4e-4)

            handles[aa] = mean[aa].plot(ax=ax[aa],
                                        y='depth',
                                        cmap=cmaps[aa],
                                        robust=True,
                                        ylim=[-250, 0],
                                        **kwargs, **pkwargs)
            plot_depths(mean, ax=ax[aa])

        for aa in axx[:-1]:
            aa.set_xlabel('')

        for aa in axx[1:]:
            aa.set_title('')

        axx[0].set_xticks([0, 90, 180, 270, 360])

        for aa in axx:
            aa.grid(True, axis='x')

        return handles

    def plot_dcl(self, region, ds='tao'):

        subset = getattr(self, ds).sel(**region)

        f, axx = plt.subplots(5, 1, constrained_layout=True, sharex=True,
                              gridspec_kw=dict(height_ratios=[1, 1, 2, 5, 5]))

        ax = dict(zip(['v', 'Q', 'dcl_KT', 'KT', 'shear'], axx))

        (np.log10(subset.KPP_diffusivity)
         .plot(ax=ax['KT'], x='time', robust=True, cmap=mpl.cm.GnBu, ylim=[-150, 0]))

        dcl_K = (subset.KPP_diffusivity.where((subset.depth < (subset.mld - 5))
                                              & (subset.depth > (subset.dcl_base + 5))))
        dcl_K = dcl_K.where(dcl_K < 1e-2)
        (dcl_K.mean('depth')
         .plot(ax=ax['dcl_KT'], x='time', yscale='log', _labels=False))
        (dcl_K.median('depth')
         .plot(ax=ax['dcl_KT'], x='time', yscale='log', _labels=False,
               ylim=[5e-4, 3e-3]))

        ax['dcl_KT'].set_ylabel('DCL $K$')

        subset.oceQnet.plot(ax=ax['Q'], x='time', _labels=False)

        subset.v.isel(depth=1).plot(ax=ax['v'], x='time', _labels=False)

        (subset.shear**2).plot(ax=ax['shear'], x='time', ylim=[-150, 0],
                               robust=True, cmap=mpl.cm.RdYlBu_r,
                               norm=mpl.colors.LogNorm(1e-6, 1e-3))

        for axx0 in [ax['KT'], ax['shear']]:
            heuc = (subset.euc_max.plot(ax=axx0, color='k', lw=1, _labels=False))
            hdcl = (subset.dcl_base.plot(ax=axx0, color='gray', lw=1, _labels=False))
            hmld = ((subset.mld - 5).plot(ax=axx0, color='k', lw=0.5, _labels=False))

        ax['v'].set_ylabel('v')
        ax['Q'].set_ylabel('$Q_{net}$')
        axx[0].set_title(ax['KT'].get_title())
        [aa.set_title('') for aa in axx[1:]]
        [aa.set_xlabel('') for aa in axx]

        ax['v'].axhline(0, color='k', zorder=-1, lw=1, ls='--')
        ax['Q'].axhline(0, color='k', zorder=-1, lw=1, ls='--')

        f.set_size_inches((8, 6))
        dcpy.plots.label_subplots(ax.values())
