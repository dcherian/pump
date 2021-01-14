import xarray as xr
import matplotlib.pyplot as plt


def validate_sst(self):

    f, ax = plt.subplots(6, 2, sharex=True, sharey=True)
    xr.concat([self.oisst, self.surface.theta], dim="type")

    self.surface.theta
