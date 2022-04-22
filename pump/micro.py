import flox
import numpy as np
from dcpy.oceans import thorpesort

from .calc import add_mixing_diagnostics


def resample_mean(ds, **kwargs):
    resampler = ds.resample(**kwargs)
    return flox.xarray.resample_reduce(resampler, func="mean")


def make_clean_dataset(ds, /, *, timeres="1H", ndepth=5):
    """
    Processes chameleon datasets

    1. Resample to hourly.
    2. Sort ε, χ, T, S, ρ by density
    3. Calculate N2, dTdz in 5m bins using density-sorted T, pden.
    4. Coarsen all fields to 5m resolution

    Parameters
    ----------
    timeres : str
        Time resolution for resampling
    ndepth : int
        Number of depth bins for coarsening
    """

    ds = ds[
        ["chi", "eps", "pres", "salt", "T", "pden", "theta", "u", "v", "eucmax", "mld"]
    ].copy(deep=True)

    ds["rhoav"] = ds.pden.mean().data + 1000
    # mask out bad values
    mask = (ds.chi < 1e-3) | (ds.eps < 1e-3)
    for var in ["chi", "eps"]:
        ds[var] = ds[var].where(mask)

    # thorpesort turbulence and CTD fields
    sort = thorpesort(
        ds[["chi", "eps", "theta", "salt", "pden"]], by="pden", core_dim="depth"
    )

    dsmean = resample_mean(ds, time=timeres)
    sortmean = resample_mean(sort, time=timeres)
    sortmean.update(dsmean[["u", "v", "pres"]])

    unsorted = (
        dsmean[["theta", "salt", "pden"]]
        .coarsen(depth=ndepth, boundary="trim")
        .mean()
        .rename(
            {
                "theta": "theta_unsorted",
                "salt": "salt_unsorted",
                "pden": "pden_unsorted",
            }
        )
    )
    reshaped = (
        sortmean.coarsen(depth=ndepth, boundary="trim")
        .construct(depth=("depth_", "window"))
        .reset_coords("depth")
    )

    avg = reshaped.mean("window")

    for var in ["eucmax", "mld"]:
        avg[var] = dsmean[var].reset_coords(drop=True)

    slopes = (
        reshaped[["theta", "pden", "u", "v"]]
        .polyfit("window", deg=1, skipna=True)
        .sel(degree=1, drop=True)
        .rename_vars(
            {
                "theta_polyfit_coefficients": "dTdz",
                "pden_polyfit_coefficients": "dρdz",
                "u_polyfit_coefficients": "dudz",
                "v_polyfit_coefficients": "dvdz",
            }
        )
    )
    avg["dTdz"] = -1 * slopes.dTdz
    avg["dTdz"] = avg.dTdz.where(np.abs(avg.dTdz) < 1)
    avg["N2"] = 9.81 / 1025 * slopes.dρdz
    avg["N2"] = avg.N2.where(avg.N2 < 5e-2)
    avg["S2"] = slopes.dudz**2 + slopes.dvdz**2
    avg["shred2"] = avg.S2 - 4 * avg.N2
    avg["Ri"] = avg.N2 / avg.S2.where(avg.S2 > 1e-6)
    avg.dTdz.attrs = {"long_name": "$T_z$", "units": "°C/m"}
    avg.N2.attrs = {"long_name": "$N^2$", "units": "s$^{-2}$"}
    avg.S2.attrs = {"long_name": "$S^2$", "units": "s$^{-2}$"}
    avg.shred2.attrs = {"long_name": "$Sh_{red}^2$", "units": "s$^{-2}$"}
    avg.Ri.attrs = {"long_name": "$Ri$"}
    avg = avg.swap_dims({"depth_": "depth"}).drop_vars("depth_")
    avg.update(unsorted)

    add_mixing_diagnostics(avg, nbins=31)

    return avg
