import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import seawater as sw
import xarray as xr
import warnings

import dcpy
import dcpy.eos
import xfilter


from numba import int64, float32, guvectorize


def merge_phase_label_period(sig, phase_0, phase_90, phase_180, phase_270, debug=False):
    """
    One version with phase=0 at points in phase_0
    One version with 360 at points in phase_0
    Then merge sensibly
    """

    if debug:
        import matplotlib.pyplot as plt

        f, ax = plt.subplots(2, 1, constrained_layout=True, sharex=True)
        sig.plot(ax=ax[0])

    phase = xr.zeros_like(sig) * np.nan
    label = xr.zeros_like(sig) * np.nan
    phase2 = phase.copy(deep=True)
    start_num = 1
    for idx, cc, ph in zip(
        [phase_0, phase_90, phase_180, phase_270], "rgbk", [0, 90, 180, 270]
    ):
        if ph == 0:
            label[idx] = np.arange(start_num, len(idx) + 1)

        if debug:
            sig[idx].plot(ax=ax[0], color=cc, ls="none", marker="o")
            ax[1].plot(
                sig.time[idx], ph * np.ones_like(idx), color=cc, ls="none", marker="o",
            )

        phase[idx] = ph
        if ph < 10:
            phase2[idx] = 360
        else:
            phase2[idx] = ph

    if not (
        np.all(np.isin(phase.dropna("time").diff("time"), [90, -270]))
        or np.all(np.isin(phase2.dropna("time").diff("time"), [90, -270]))
    ):
        import warnings

        warnings.warn("Secondary peaks detected!")

    phase = phase.interpolate_na("time", method="linear")
    phase2 = phase2.interpolate_na("time", method="linear")

    dpdt = phase.differentiate("time")

    phase_new = xr.where(
        (phase2 >= 270) & (phase2 < 360) & (phase < 270) & (dpdt <= 0), phase2, phase,
    )

    if debug:
        phase_new.plot(ax=ax[1], color="C0", zorder=-1)

    label = label.ffill("time")

    # periods don't necessarily start with phase = 0
    phase_no_period = np.logical_and(~np.isnan(phase_new), np.isnan(label))
    label.values[phase_no_period.values] = 0

    if np.any(label == 0):
        label += 1

    if debug:
        ax2 = ax[1].twinx()
        label.plot(x="time", ax=ax2, color="k", lw=0.5)

    phase_new.name = "tiw_phase"
    label.name = "period"
    return phase_new, label


def calc_reduced_shear(data):
    """
    Estimate reduced shear for a dataset. Dataset must contain
    'u', 'v', 'depth', 'dens'.
    """

    data["S2"] = data.u.differentiate("depth") ** 2 + data.v.differentiate("depth") ** 2
    data["S2"].attrs["long_name"] = "$S^2$"
    data["S2"].attrs["units"] = "s$^{-2}$"

    data["shear"] = np.sqrt(data.S2)
    data["shear"].attrs["long_name"] = "|$u_z$|"
    data["shear"].attrs["units"] = "s$^{-1}$"

    # data['N2'] = (9.81 * 1.7e-4 * data.theta.differentiate('depth')
    #              - 9.81 * 7.6e-4 * data.salt.differentiate('depth'))

    data["N2"] = -9.81 / 1025 * data.dens.differentiate("depth")
    data["N2"].attrs["long_name"] = "$N^2$"
    data["N2"].attrs["units"] = "s$^{-2}$"

    data["shred2"] = data.S2 - 4 * data.N2
    data.shred2.attrs["long_name"] = "Reduced shear$^2$"
    data.shred2.attrs["units"] = "$s^{-2}$"

    data["Ri"] = data.N2 / data.S2
    data.Ri.attrs["long_name"] = "Ri"
    data.Ri.attrs["units"] = ""

    return data


def _get_max(var, dim="depth"):

    # return((xr.where(var == var.max(dim), var[dim], np.nan))
    #       .max(dim))

    coords = dict(var.coords)
    coords.pop(dim)

    dims = list(var.dims)
    del dims[var.get_axis_num(dim)]

    # non_nans = var
    # for dd in dims:
    #    non_nans = non_nans.dropna(dd, how="all")
    argmax = var.fillna(-123456).argmax(dim)
    argmax = argmax.where(argmax != 0)

    new_coords = dict(var.coords)
    new_coords.pop(dim)

    da = xr.DataArray(argmax.data.squeeze(), dims=dims, coords=new_coords).compute()
    return (
        var[dim][da.fillna(0).astype(int)]
        .drop(dim)
        .reindex_like(var)
        .where(da.notnull())
    )


def get_euc_max(u, kind="model"):
    """ Given a u field, returns depth of max speed i.e. EUC maximum. """

    if kind == "data":
        u = u.fillna(-100)
    euc_max = _get_max(u, "depth")

    euc_max.attrs["long_name"] = "Depth of EUC max"
    euc_max.attrs["units"] = "m"

    return euc_max


def get_dcl_base_shear(data):
    """
    Estimates base of the deep cycle layer as the depth of max total shear squared.

    References
    ----------

    Inoue et al. (2012)
    """

    if "S2" in data:
        s2 = data.S2
    elif "shear" in data:
        s2 = data["shear"] ** 2
    else:
        if "u" not in data and "v" not in data:
            raise ValueError("S2 or shear, or (u,v) not found in provided dataset.")
        s2 = data.u.differentiate("depth") ** 2 + data.v.differentiate("depth") ** 2

    if "euc_max" not in data:
        euc_max = get_euc_max(data.u)
    else:
        euc_max = data.euc_max

    dcl_max = _get_max(
        s2.where(s2.depth < -20, 0).where(s2.depth > euc_max, 0), "depth"
    )

    dcl_max.attrs["long_name"] = "DCL Base (shear)"
    dcl_max.attrs["units"] = "m"
    dcl_max.attrs["description"] = "Depth of maximum total shear squared (above EUC)"

    return dcl_max


def get_dcl_base_Ri(data):
    """
    Estimates base of the deep cycle layer as max depth where Ri <= 0.25.

    References
    ----------

    Lien et. al. (1995)
    Pham et al (2017)
    """

    if "Ri" not in data:
        raise ValueError("Ri not in provided dataset.")

    if "eucmax" not in data:
        euc_max = get_euc_max(data.u)
    else:
        euc_max = data.eucmax

    if np.any(data.depth > 0):
        raise ValueError("depth > 0!")

    dcl_max = data.depth.where((data.Ri > 0.5)).max("depth")

    dcl_max.attrs["long_name"] = "DCL Base (Ri)"
    dcl_max.attrs["units"] = "m"
    dcl_max.attrs["description"] = "Deepest depth above EUC where Ri=0.25"

    return dcl_max


def get_euc_transport(u):
    euc = u.where(u > 0).sel(latitude=slice(-3, 3, None))
    euc.values[np.isnan(euc.values)] = 0
    euc = euc.integrate("latitude") * 0.1

    return euc


def calc_tao_ri(adcp, temp, dim="depth"):
    """
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
    """

    V = adcp[["u", "v"]].sortby(dim).interpolate_na(dim)
    S2 = V["u"].differentiate(dim) ** 2 + V["v"].differentiate(dim) ** 2

    T = temp.sortby(dim).interpolate_na(dim, "linear")

    if "time" in T.dims and not T.time.equals(V.time):
        T = temp.sel(time=V.time)

    if not T[dim].equals(V[dim]):
        T = T.interp({dim: V[dim]})

    # the calculation is sensitive to using sw.alpha! can't just do 1.7e-4
    N2 = 9.81 * dcpy.eos.alpha(35, T, T.depth) * T.differentiate(dim)
    Ri = (N2 / S2).where((N2 > 1e-7) & (S2 > 1e-10))

    Ri.attrs["long_name"] = "Ri"
    Ri.name = "Ri"

    return Ri


def kpp_diff_depth(obj, debug=False):
    """
    Determine KPP mixing layer depth by searching for the first depth where
    diffusivity is less than the diffusivity at depth level 1
    (depth level 0 is NaN).
    """
    z0 = 2
    depth = xr.where(
        (obj.isel(depth=slice(z0, None)) < obj.isel(depth=z0)),
        obj.depth.isel(depth=slice(z0, None)),
        np.nan,
    ).max("depth")

    if debug:
        obj.plot(ylim=[-120, 0], y="depth", xscale="log", marker=".")
        dcpy.plots.liney(depth, color="r")

    depth.name = "kpp_diff_mld"
    depth.attrs["long_name"] = "KPP MLD from diffusivity"
    depth.attrs["units"] = "m"

    return depth


def get_kpp_mld(subset, debug=False):
    """
    Given subset.dens, subset.u, subset.v, estimate MLD as shallowest depth
    where KPP Rib < 0.05.
    """

    import dask

    b = (-9.81 / 1025) * subset.dens
    V = np.hypot(subset.u, subset.v)
    Rib = (b.isel(depth=0) - b) * (-b.depth) / (V.isel(depth=0) - V) ** 2
    kpp_mld = xr.where(Rib > 0.1, Rib.depth, np.nan).max("depth")

    kpp_mld.name = "kpp_mld"
    kpp_mld.attrs["long_name"] = "KPP bulk Ri MLD"
    kpp_mld.attrs["units"] = "m"

    if debug:
        assert "time" not in b.dims
        import dcpy

        # dRib.plot.line(y="depth")
        Rib.plot.line(y="depth", ylim=[-120, 0], xlim=[-1, 1])
        dcpy.plots.liney(kpp_mld)
        dcpy.plots.linex([0, 0.3])

    return kpp_mld


def get_mld(dens):
    """
    Given density field, estimate MLD as depth where drho > 0.01 and N2 > 2e-5.
    # Interpolates density to 1m grid.
    """
    if not isinstance(dens, xr.DataArray):
        raise ValueError(f"Expected DataArray, received {dens.__class__.__name__}")

    # densi = dens  # .interp(depth=np.arange(0, -200, -1))
    drho = dens - dens.isel(depth=0)
    N2 = -9.81 / 1025 * dens.differentiate("depth")

    thresh = xr.where((np.abs(drho) > 0.015) & (N2 > 1e-5), drho.depth, np.nan)
    mld = thresh.max("depth")

    mld.name = "mld"
    mld.attrs["long_name"] = "MLD"
    mld.attrs["units"] = "m"
    mld.attrs["description"] = (
        "Interpolate density to 1m grid. "
        "Search for max depth where "
        " |drho| > 0.01 and N2 > 1e-5"
    )

    return mld


def tiw_avg_filter_v(v):
    v = xfilter.lowpass(
        v.sel(depth=slice(-10, -80)).mean("depth"),
        coord="time",
        freq=1 / 10.0,
        cycles_per="D",
        method="pad",
        gappy=False,
        num_discard=0,
    )

    if v.count() == 0:
        raise ValueError("No good data in filtered depth-averaged v.")

    v.attrs["long_name"] = "v: (10, 80m) avg, 10d lowpass"

    return v


def _find_phase_single_lon(sig, debug=False):

    if np.sum(sig.shape) == 0:
        out = xr.Dataset()
        out["tiw_phase"] = sig.copy()
        out["period"] = sig.copy()
        return out

    sig = sig.squeeze()
    peak_kwargs = {"prominence": 0.1}
    phase_90 = sp.signal.find_peaks(-sig, **peak_kwargs)[0]
    phase_270 = sp.signal.find_peaks(sig, **peak_kwargs)[0]
    phase_180 = []
    for i90, i270 in zip(phase_90, phase_270):
        sig180 = (sig[i90] + sig[i270]) / 2
        phase_180.append(np.abs(sig[i90:i270] - sig180).argmin().values + i90)

    phase_0 = []
    for i90, i270 in zip(phase_90[1:], phase_270):
        sig0 = (sig[i90] + sig[i270]) / 2
        phase_0.append(np.abs(sig[i270:i90] - sig0).argmin().values + i270)

    phase, period = merge_phase_label_period(
        sig, phase_0, phase_90, phase_180, phase_270, debug=debug,
    )

    return xr.merge([phase, period]).expand_dims("longitude")


def get_tiw_phase_sst(sst, debug=False):

    sstfilt = xfilter.lowpass(
        sst.sel(latitude=slice(0, 5)).mean("latitude"),
        coord="time",
        freq=1 / 15,
        cycles_per="D",
        num_discard=0,
    )

    output = sstfilt.map_blocks(_find_phase_single_lon, kwargs={"debug": debug})

    return output["tiw_phase"], output["period"]


def get_tiw_phase_v(v, debug=False):
    """
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
    """

    v = tiw_avg_filter_v(v)

    if v.ndim == 1:
        v = v.expand_dims("new_dim").copy()
        unstack = False
    elif v.ndim > 2:
        unstack = True
        v = v.stack({"stacked": set(v.dims) - set(["time"])})
    else:
        unstack = False

    dvdt = v.differentiate("time")

    zeros_da = xr.where(
        np.abs(v) < 1e-2,
        xr.DataArray(
            np.arange(v.shape[v.get_axis_num("time")]),
            dims=["time"],
            coords={"time": v.time},
        ),
        np.nan,
    )

    assert v.ndim == 2
    dim2 = list(set(v.dims) - set(["time"]))[0]

    phases = []
    labels = []
    peak_kwargs = {"prominence": 0.02}

    for dd in v[dim2]:
        vsub = v.sel({dim2: dd})
        if debug:
            f, ax = plt.subplots(2, 1, sharex=True, constrained_layout=True)

            vsub.plot(ax=ax[0], x="time")
            dcpy.plots.liney(0, ax=ax[0])

        zeros = zeros_da.sel({dim2: dd}).dropna("time").values.astype(np.int32)

        zeros_unique = zeros[np.insert(np.diff(zeros), 0, 100) > 1]

        phase_0 = sp.signal.find_peaks(vsub, **peak_kwargs)[0]
        phase_90 = zeros_unique[
            np.nonzero(dvdt.sel({dim2: dd}).values[zeros_unique] < 0)[0]
        ]
        phase_180 = sp.signal.find_peaks(-vsub, **peak_kwargs)[0]
        phase_270 = zeros_unique[
            np.nonzero(dvdt.sel({dim2: dd}).values[zeros_unique] > 0)[0]
        ]

        # 0 phase must be positive v
        idx = phase_0
        if not np.all(vsub[idx] > 0):
            idx = np.where(vsub[idx] > 0, idx, np.nan)
            idx = idx[~np.isnan(idx)].astype(np.int32)

        # 180 phase must be negative v
        idx = phase_180
        if not np.all(vsub[idx] < 0):
            idx = np.where(vsub[idx] < 0, idx, np.nan)
            idx = idx[~np.isnan(idx)].astype(np.int32)

        phase_new, label = merge_phase_label_period(
            vsub, phase_0, phase_90, phase_180, phase_270
        )

        vamp = np.abs(
            vsub.isel(
                time=np.sort(
                    np.hstack([phase_0[0], phase_90[0], phase_180[0], phase_270[0]])
                )
            ).diff("time", label="lower")
        )
        vampf = vamp.reindex(time=phase_new.time).ffill("time")
        phase_new = phase_new.where(vampf > 0.1)

        if debug:
            # vampf.plot.step(ax=ax[0])
            phase_new.plot(ax=ax[1])
            dcpy.plots.liney([0, 90, 180, 270, 360], ax=ax[1])
            ax2 = ax[1].twinx()
            (label.ffill("time").plot(ax=ax2, x="time", color="k"))
            ax[0].set_xlabel("")
            ax[1].set_title("")
            ax[1].set_ylabel("TIW phase [deg]")

        for dd in set(list(phase_new.coords)) - set(["time"]):
            phase_new = phase_new.expand_dims(dd)
            label = label.expand_dims(dd)

        phases.append(phase_new)
        labels.append(label)

    phase = xr.merge(phases).to_array().squeeze()
    phase.attrs["long_name"] = "TIW phase"
    phase.name = "tiw_phase"
    phase.attrs["units"] = "deg"

    label = xr.merge(labels).to_array().squeeze().ffill("time")
    label.name = "period"

    phase = xr.merge([phase, label])

    if unstack:
        # lost stack information earlier; re-assign that
        phase["stacked"] = v["stacked"]
        phase = phase.unstack("stacked")

    # phase['period'] = phase.period.where(~np.isnan(phase.tiw_phase))

    # get rid of 1 point periods
    mask = phase.tiw_phase.groupby(phase.period).count() == 1
    drop_num = mask.period.where(mask, drop=True).values
    phase["period"] = phase["period"].where(np.logical_not(phase.period.isin(drop_num)))

    return phase


def estimate_euc_depth_terms(ds, inplace=True):

    # ds.load()

    if not inplace:
        ds = ds.copy()

    surface = {"depth": -25, "method": "nearest"}

    ds["h"] = ds.eucmax - surface["depth"]
    ds["h"].attrs["long_name"] = "$h$"

    euc = ds.where(ds.depth == ds.eucmax).max("depth")

    if "u" in ds:
        ds["us"] = ds.u.ffill("depth").sel(**surface)
        ds["ueuc"] = euc.u
        # ds["ueuc"] = ds.u.interp(
        #    depth=ds.eucmax, longitude=ds.longitude, method="linear"
        # )
        ds["du"] = ds.us - ds.ueuc
        ds.du.attrs["long_name"] = "$\Delta$u"

    if "dens" in ds:
        ds["dens_euc"] = ds.dens.interp(
            depth=ds.eucmax, longitude=ds.longitude, method="linear"
        )
        # ds["dens_euc"] = euc.dens
        ds["b"] = ds.dens * -9.81 / ds.dens_euc
        ds["bs"] = ds.b.ffill("depth").sel(**surface)
        ds["beuc"] = -9.81 * xr.ones_like(ds.bs)

        ds["db"] = ds.bs - ds.beuc
        ds.db.attrs["long_name"] = "$\Delta$b"

    if "db" in ds and "du" in ds and "h" in ds:
        ds = estimate_Rib(ds)
    return ds


def estimate_Rib(ds):
    with xr.set_options(keep_attrs=False):
        ds["Rib"] = ds.db * np.abs(ds.h) / (ds.du ** 2)
    return ds


get_tiw_phase = get_tiw_phase_v


def get_dcl_base_dKdt(K, threshold=0.03, debug=False):
    def minmax(group):
        mx = group.where(group > 0).max("time")
        mn = group.where(group < 0).min("time")

        coord = xr.DataArray(["min", "max"], dims="minmax", name="minmax")
        return xr.concat([mn, mx], coord)

    dKdt = K.differentiate("time")
    grouped = (
        dKdt.groupby(dKdt.time.dt.floor("D")).apply(minmax).rename({"floor": "time"})
    )
    amp = grouped.diff("minmax").squeeze()
    amp_cleaned = amp.sortby("depth").chunk({"depth": -1}).interpolate_na("depth")

    dcl = (
        amp_cleaned.depth.where(amp_cleaned / amp_cleaned.max("depth") < threshold)
        .max("depth")
        .reindex(time=dKdt.time, method="ffill")
    )

    if debug:
        plt.figure()
        (
            dKdt
            .groupby("time.day").plot(
                col="day",
                col_wrap=6,
                x="time",
                # sharey=True,
                ylim=(-100, 0),
                robust=True,
            )
        )
        dcl.plot(x="time")

        plt.figure()
        fg = (
            (amp_cleaned / amp_cleaned.max("depth"))
            # .sel(depth=-20, method="nearest")
            .plot(
                col="time",
                col_wrap=6,
                y="depth",
                # sharey=True,
                ylim=(-100, 0),

            )
        )

    return dcl
