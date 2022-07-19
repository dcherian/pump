import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from .calc import get_euc_max

SPONGE_AMP = 0.000001
MLDTHRESH = 0.00015  # buoyancy


def derivative(da, dz):
    ddz = xr.zeros_like(da)
    ddz.attrs = {}
    ddz.data[1:-1, :] = (da.data[2:, :] - da.data[:-2, :]) / 2 / dz.data
    ddz.data[0, :] = ddz.data[1, :]
    ddz.data[-1, :] = ddz.data[-2, :]
    return ddz


def read_les_file(fname):
    ds = xr.open_dataset(fname, decode_cf=False)
    return preprocess_les_dataset(ds)


def add_ancillary_variables(ds):
    """Add variables that I can calculate with both mooring and domain-average files."""
    ds["buoy"] = -9.81 * ds.rho / ds.rho0
    ds.buoy.attrs["standard_name"] = "sea_water_buoyancy"

    mask = ds.buoy.isel(z=slice(1, -1)) < (
        ds.buoy.isel(z=slice(-20, -1)).mean("z") - MLDTHRESH
    )
    ds["mld"] = ds.z.where(mask).max("z")
    ds["mld"].attrs["standard_name"] = "ocean_mixed_layer_thickness"

    ds["dudz"] = derivative(ds.cf["sea_water_x_velocity"], ds.dz)
    ds["dvdz"] = derivative(ds.cf["sea_water_y_velocity"], ds.dz)
    ds["dudzim"] = ds.dudz + 1j * ds.dvdz

    ds["dTdz"] = derivative(ds.cf["sea_water_potential_temperature"], ds.dz)
    ds["dSdz"] = derivative(ds.cf["sea_water_salinity"], ds.dz)


def preprocess_les_dataset(ds):
    ds = ds.cf.guess_coord_axis()  # .set_coords(["alpha", "beta", "T0", "S0"])
    for var in ds:
        ds[var].attrs["_FillValue"] = 9.969209968386869e36
    ds = xr.decode_cf(ds)

    ds["z"].data[-1] = 0
    ds.coords["dz"] = 0.5
    ds["zc"] = ("zc", (ds.z.data[:-1] + ds.z.data[1:]) / 2)

    if "tempw" not in ds:
        # mooring file
        ds["u"].attrs["standard_name"] = "sea_water_x_velocity"
        ds["v"].attrs["standard_name"] = "sea_water_y_velocity"
        ds["temp"].attrs["standard_name"] = "sea_water_potential_temperature"
        ds["salt"].attrs["standard_name"] = "sea_water_salinity"
        ds["rho"] = ds.rho0 * (
            1 - ds.alpha * (ds.temp - ds.T0) - ds.beta * (ds.salt - ds.S0)
        )
        ds.rho.attrs["standard_name"] = "sea_water_potential_density"

        add_ancillary_variables(ds)
        return ds

    ds["ume"].attrs["standard_name"] = "sea_water_x_velocity"
    ds["vme"].attrs["standard_name"] = "sea_water_y_velocity"
    ds["rho"] = ds.rho0 * (
        1 - ds.alpha * (ds.tempme - ds.T0) - ds.beta * (ds.saltme - ds.S0)
    )
    ds.rho.attrs["standard_name"] = "sea_water_potential_density"
    ds["tempme"].attrs["standard_name"] = "sea_water_potential_temperature"
    ds["saltme"].attrs["standard_name"] = "sea_water_salinity"

    add_ancillary_variables(ds)

    ds["wb"] = 9.81 * (ds.alpha * ds.tempw + ds.beta * ds.saltw)

    ds["kappadbdz"] = 9.81 * (ds.alpha * ds.kappadtdz + ds.beta * ds.kappadsdz)

    ds["kappadbdztop"] = 9.81 * (ds.alpha * ds.kappadtdztop + ds.beta * ds.kappadsdztop)

    ds["dBdtSOLAR"] = 9.81 * ds.alpha * ds.dTdtSOLAR
    ds["dBdtsolarsum"] = ds.dBdtSOLAR.isel(z=slice(1, -1)).sum("z") * ds.dz
    ds["dTdtsolarsum"] = ds.dTdtSOLAR.isel(z=slice(1, -1)).sum("z") * ds.dz

    # positive means downwards transport of momentum
    ds["Fim"] = ds.nududz - ds.uw + 1j * (ds.nudvdz - ds.vw)
    ds["Fimtop"] = ds.nududztop + 1j * (ds.nudvdztop)

    # note Fb includes subgrid scale temperature flux, kappadTdz,
    # resolved temperature flux -tempw, subgridscale salinity flux kappadSdz,
    # and resolved salinity flux -saltw
    ds["Fb"] = ds.kappadbdz - ds.wb
    ds["FT"] = ds.kappadtdz - ds.tempw
    ds["FS"] = ds.kappadsdz - ds.saltw

    ds["KM"] = np.real(ds.Fim * np.conj(ds.dudzim)) / (ds.dudzim * np.conj(ds.dudzim))
    ds["Kb"] = (ds.Fb * ds.N2) / ((ds.N2) ** 2)
    ds["KT"] = (ds.FT * ds.dTdz) / (ds.dTdz**2)
    ds["KS"] = (ds.FS * ds.dSdz) / (ds.dSdz**2)
    ds["SHEARPROD"] = np.real(ds.Fim * np.conj(ds.dudzim))

    ds["epsilon"] = ds.epsilon.where((ds.epsilon < 1) & (ds.epsilon > 0))
    ds["chi"] = ds.FT * ds.dTdz

    ds["tke"] = 0.5 * (ds.urms**2 + ds.vrms**2 + ds.wrms**2)

    # add an Rif variable to the data structure:
    # we don't trust Rif where turbulence is weak; this is just
    # parameterization
    # let's just crudely remote regions of low epsilon with a fixed threshold
    ds["Rif"] = ds.Fb / (ds.epsilon.where(ds.epsilon > 1e-8) + ds.Fb)
    ds["Rif"].attrs = {"long_name": "$Ri_f$"}

    ds["Reb"] = ds.epsilon / (1e-6 * ds.N2)
    ds["Reb"].attrs = {"long_name": "$Re_b$"}

    ds["Jq"] = 1025 * 4200 * ds.kappadtdz
    ds.Jq.attrs = {"long_name": "$J_q$", "units": "W/m^2"}

    ds["eucmax"] = get_euc_max(ds.ume)

    return ds


def write_to_txt(ds, outdir, prefix, interleave, t0=None):

    f = debug_plots(ds.drop_vars(["XG", "YG"]))
    f.suptitle(prefix[:-1], y=1.02)
    f.savefig(f"{outdir}/{prefix}_plot.png", dpi=150)

    if interleave:
        assert t0 is not None
        ds = interleave_time(ds, t0)

    ds = ds.cf.transpose("time", ...)
    for var in ds:
        ds[var].data.ravel().tofile(
            f"{outdir}/{prefix}{var}.txt", sep="\n", format="%15.7f"
        )


def interleave_time(ds, t0):
    """Interleaves time with values. For colpsh and colfrc files."""

    time = ds.cf["time"]

    assert time.ndim == 1

    ds.coords["reltime"] = (
        time.dims[0],
        (ds.cf["time"].data - t0).astype("timedelta64[s]").astype(float),
    )

    def _interleave(da):
        stacked = np.hstack([da.reltime.data, da.squeeze().data])
        return xr.DataArray(stacked, dims="zt")

    return ds.map(lambda x: x.cf.groupby("time", squeeze=False).apply(_interleave))


def interpolate(ds, newz):
    """Interpolate to les Z grid"""

    # dcpy.interpolate.UnivariateSpline(subset, "z_rho").interp(newz.values)
    interped = ds.cf.interp(
        Z=newz.values, method="slinear", kwargs={"fill_value": "extrapolate"}
    ).compute()

    # check that there are no NaNs
    for k, v in interped.isnull().sum().items():
        assert v.data == 0, "NaNs detected after interpolation"

    return interped


def debug_plots(ds):

    varnames = sorted(set(ds.data_vars) - {"time"})

    if "time" in ds.cf.dims:
        sharex = True
    else:
        sharex = False

    f, ax = plt.subplots(len(varnames), 1, sharex=sharex, constrained_layout=True)
    for var, axis in zip(varnames, ax.squeeze()):
        if var == "time":
            continue
        if (
            ds[var].ndim == 1
            and "time" not in ds[var].dims
            and "time_index" not in ds[var].dims
        ):
            ds[var].plot(y="RC", ax=axis)
        elif "time" in ds[var].dims:
            ds[var].cf.plot(x="time", ax=axis)
        else:
            ds[var].cf.plot(x="time_index", ax=axis)

    f.set_size_inches((6, 8))

    return f
