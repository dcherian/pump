import matplotlib.pyplot as plt


def plot_depths(ds, ax=None, **kwargs):

    if ax is None:
        ax = plt.gca()

    if 'euc_max' in ds:
        heuc = (ds.euc_max.plot.line(ax=ax, color='k', lw=1, _labels=False, **kwargs))

    if 'dcl_base' in ds:
        hdcl = (ds.dcl_base.plot.line(ax=ax, color='gray', lw=1, _labels=False, **kwargs))

    if 'mld' in ds:
        hmld = ((ds.mld).plot.line(ax=ax, color='k', lw=0.5, _labels=False, **kwargs))
