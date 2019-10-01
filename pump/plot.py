import dcpy.plots
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def plot_depths(ds, ax=None, **kwargs):

    if ax is None:
        ax = plt.gca()

    if 'euc_max' in ds:
        heuc = (ds.euc_max.plot.line(ax=ax, color='k', lw=1, _labels=False, **kwargs))

    if 'dcl_base' in ds:
        hdcl = (ds.dcl_base.plot.line(ax=ax, color='gray', lw=1, _labels=False, **kwargs))

    if 'mld' in ds:
        hmld = ((ds.mld).plot.line(ax=ax, color='k', lw=0.5, _labels=False, **kwargs))


def plot_bulk_Ri_diagnosis(ds, f=None, ax=None, **kwargs):
    '''
    Estimates fractional contributions of various terms to bulk Richardson
    number.
    '''

    def plot_ri_contrib(ax1, ax2, v, factor=1, **kwargs):
        # Better to call differentiate on log-transformed variable
        # This is a nicer estimate of the gradient and is analytically equal
        per = factor * np.log(np.abs(v)).differentiate('longitude')
        hdl = per.plot(ax=ax2, x='longitude',
                       label=f'{factor}/{v.name} $∂_x${v.name}',
                       add_legend=False,
                       **kwargs)
        v.plot(ax=ax1, x='longitude', **kwargs)
        ax1.set_xlabel('')
        ax1.set_title('')

        return per, hdl

    if f is None and ax is None:
        f, axx = plt.subplots(7, 1, constrained_layout=True, sharex=True,
                              gridspec_kw={
                                  'height_ratios': [1, 1, 1, 1, 1, 1, 2]
                              })
        ax = dict(zip(['Ri', 'h', 'du', 'db', 'u', 'b', 'contrib'], axx))
        add_legend = True
    else:
        add_legend = False

    colors = dict({'us': 'C0', 'ueuc': 'C1', 'bs': 'C0', 'beuc': 'C1',
                   'Ri': 'C0', 'h': 'C1', 'du': 'C2', 'db': 'C3'})

    factor = dict(zip(ax.keys(), [1, 1, -2, 1]))
    rhs = xr.zeros_like(ds.bs)
    per = dict()
    for var in ax.keys():
        if var not in ['u', 'b', 'contrib']:
            per[var], hdl = plot_ri_contrib(
                ax[var], ax['contrib'], ds[var], factor[var],
                color=colors[var], **kwargs)
            if var != 'Ri':
                rhs += per[var]
            else:
                ri = per[var]
                if 'marker' not in kwargs:
                    hdl[0].set_marker('o')

    for vv in ['u', 'b']:
        for vvar in ['s', 'euc']:
            var = vv + vvar
            if vvar == 'euc':
                factor = -1
                prefix = '-'
            else:
                factor = 1
                prefix = ''
            (factor*ds[var].differentiate('longitude')).plot(
                ax=ax[vv],
                label=f'{prefix}$∂_x {vv}_{{{vvar}}}$', # label=f'$∂_x{vv}_{{{vvar}}}$',
                color=colors[var],
                **kwargs)
            if add_legend:
                ax[vv].legend(ncol=2)

            dcpy.plots.liney(0, ax[vv])
            ax[vv].set_title('')
            ax[vv].set_xlabel('')
            ax[vv].set_ylabel('')

    ax['u'].set_ylim([-0.04, 0.04])
    ax['b'].set_ylim([-0.0007, 0.0007])

    ax['du'].set_ylim([-1.3, -0.3])
    ax['db'].set_ylim([0.005, 0.05])

    ax['Ri'].set_ylabel('Ri$_b =  Δbh/Δu²$')
    ax['Ri'].set_yscale('log')
    ax['Ri'].set_yticks([0.25, 0.5, 1, 5, 10])
    ax['Ri'].grid(True)

    rhs.plot(ax=ax['contrib'], x='longitude', color='k', label='RHS', **kwargs,
             add_legend=False)
    if add_legend:
        ax['contrib'].legend(ncol=5)
        dcpy.plots.liney(0, ax=ax['contrib'])
    ax['contrib'].set_ylabel('Fractional changes')
    ax['contrib'].set_title('')
    ax['contrib'].set_ylim([-0.15, 0.1])

    name = ds.attrs['name']
    if add_legend:
        ax['Ri'].set_title(f"latitude = 0, {name} dataset")
    else:
        ax['Ri'].set_title(f"latitude = 0")

    f.set_size_inches(8, 10)

    # xr.testing.assert_allclose(ri, rhs)

    return f, ax
