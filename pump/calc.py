import dask
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seawater as sw
import xarray as xr
import warnings

import dcpy
import dcpy.eos
import xfilter


from numba import int64, float32, guvectorize

from . import KPP


def ddx(a):
    return a.differentiate("longitude") / 110e3


def ddy(a):
    return a.differentiate("latitude") / 110e3


def ddz(a):
    return a.differentiate("depth")


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

    print("calc uz")
    data["uz"] = data.u.differentiate("depth")

    print("calc vz")
    data["vz"] = data.v.differentiate("depth")

    print("calc S2")
    data["S2"] = data.uz ** 2 + data.vz ** 2
    data["S2"].attrs["long_name"] = "$S^2$"
    data["S2"].attrs["units"] = "s$^{-2}$"

    data["shear"] = np.sqrt(data.S2)
    data["shear"].attrs["long_name"] = "|$u_z$|"
    data["shear"].attrs["units"] = "s$^{-1}$"

    # data['N2'] = (9.81 * 1.7e-4 * data.theta.differentiate('depth')
    #              - 9.81 * 7.6e-4 * data.salt.differentiate('depth'))

    print("calc N2")
    data["N2"] = -9.81 / 1025 * data.dens.differentiate("depth")
    data["N2"].attrs["long_name"] = "$N^2$"
    data["N2"].attrs["units"] = "s$^{-2}$"

    print("calc shred2")
    data["shred2"] = data.S2 - 4 * data.N2
    data.shred2.attrs["long_name"] = "Reduced shear$^2$"
    data.shred2.attrs["units"] = "$s^{-2}$"

    print("Calc Ri")
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


def get_dcl_base_Ri(data, mld=None, eucmax=None):
    """
    Estimates base of the deep cycle layer as max depth where Ri <= 0.25.

    References
    ----------

    Lien et. al. (1995)
    Pham et al (2017)
    """

    if "Ri" not in data:
        raise ValueError("Ri not in provided dataset.")

    # if "eucmax" not in data:
    #    euc_max = get_euc_max(data.u)
    # else:
    #    euc_max = data.eucmax

    if np.any(data.depth > 0):
        raise ValueError("depth > 0!")

    if "mld" in data and mld is None:
        mld = data.mld
    elif mld is None:
        raise ValueError("Please provide mld in dataset or as kwarg")

    Ric = 0.54

    dcl_max_1 = data.depth.where(
        data.Ri.where((data.depth <= mld) & (data.depth > -150)) > Ric
    ).max("depth")

    mask_2 = (data.depth <= (mld - 25)) & (np.abs(data.Ri - Ric) < 0.1)
    if eucmax is not None:
        mask_2 = mask_2 & (data.depth > eucmax)
    dcl_max_2 = data.Ri.depth.where(mask_2)
    counts = dcl_max_2.count("depth")
    dcl_max_2 = dcl_max_2.fillna(-12345).max("depth")

    # cum_Ri = data.Ri.where(mask_2).cumsum("depth") / data.Ri.depth.copy(
    #    data=np.arange(1, data.sizes["depth"] + 1)
    # )
    # dcl_max = data.Ri.depth.where(cum_Ri < Ric).min("depth")

    maybe_too_shallow = np.abs(dcl_max_1 - mld) < 25
    dcl_max = xr.where(maybe_too_shallow & (counts > 0), dcl_max_2, dcl_max_1)

    mask_3 = (data.depth <= mld) & (data.depth >= dcl_max)
    if eucmax is not None:
        mask_3 = mask_3 & (data.depth > eucmax)
    median_dcl_Ri = data.Ri.where(mask_3).median("depth")
    median_Ri_too_high = median_dcl_Ri > (Ric + 0.05)
    dcl_max = dcl_max.where(np.logical_not(median_Ri_too_high), mld)
    # dcl_max = dcl_max.ffill("time")

    dcl_max.attrs["long_name"] = "DCL Base (Ri)"
    dcl_max.attrs["units"] = "m"
    dcl_max.attrs["description"] = f"Deepest depth above EUC where Ri={Ric}"

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

    # gcm1 is 1m
    # densi = dcpy.interpolate.pchip(dens, "depth", np.arange(0, -200, -1))
    densi = dens  # .interp(depth=np.arange(0, -200, -1))
    drho = densi - densi.isel(depth=0)
    N2 = -9.81 / 1025 * densi.differentiate("depth")

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


def crossings_nonzero_all(data):
    pos = data > 0
    npos = ~pos
    return ((pos[:-1] & npos[1:]) | (npos[:-1] & pos[1:])).nonzero()[0]


def fix_phase_using_sst_front(gr, debug=False):
    # print(gr.period[0].values)
    ph = gr.tiw_phase
    indexes, properties = sp.signal.find_peaks(
        gr.where(ph < 180, drop=True).values, prominence=0.3
    )

    sorted_idx = np.argsort(properties["prominences"])
    # print(properties["prominences"][sorted_idx])
    if len(indexes) > 0:
        # use most prominent peak
        # print(gr.time[indexes[sorted_idx]])
        tfix = gr.time[indexes[sorted_idx[-1]]]
        # print(tfix.values)

        if debug:
            plt.figure()
            gr.plot()
            dcpy.plots.linex(gr.time[indexes])
            dcpy.plots.linex(tfix, color="r")
            plt.title("fixing phase using front")

        last = ph[-1]
        ph = ph.where(ph.isin([0, 180, 270]))
        ph[-1] = last
        ph.loc[tfix] = 90
        ph = ph.interpolate_na("time")
    return ph


def _find_phase_single_lon(sig, algo_0_180="zero-crossing", debug=False):

    ds = sig
    sig = ds["sst"]
    grad = ds["grad"].squeeze()

    nlon = sig.sizes["longitude"]
    if nlon > 1:
        raise ValueError(
            f"Length in longitude dimension must be 1. Received {nlon} instead"
        )

    if np.sum(sig.shape) == 0:
        out = xr.Dataset()
        out["tiw_phase"] = sig.copy()
        out["period"] = sig.copy()
        out["tiw_ptp"] = sig.copy()
        return out

    sig = sig.squeeze()

    prominences = {-110: 0.3, -125: 0.1, -140: 0.1, -155: 0.1}
    peak_kwargs = {"prominence": prominences[sig.longitude.squeeze().values.item()]}

    if debug:
        plt.figure()
        sig.plot()

    phase_90 = sp.signal.find_peaks(-sig, **peak_kwargs)[0]
    phase_270 = sp.signal.find_peaks(sig, **peak_kwargs)[0]

    if debug:
        # print(phase_90)
        # print(phase_270)
        dcpy.plots.linex(sig.time[phase_90])
        dcpy.plots.linex(sig.time[phase_270])

    if phase_90[0] > phase_270[0]:
        phase_90 = np.insert(phase_90, 0, 0)
        inserted_0 = True
    else:
        inserted_0 = False

    phase_180 = []
    phase_0 = []

    if algo_0_180 == "zero-crossing":
        # get rid of saddle points that may not be ideal for zero-crossing detection
        # replace with zeros and then use the mean (mean selection happens later)
        dsig = np.abs(sig.differentiate("time"))
        thresh = dsig.quantile(q=0.3, dim="time")
        for_zeros = sig.where(
            ~((np.abs(sig) < np.abs(sig).quantile(q=0.1, dim="time")) & (dsig < thresh))
        ).fillna(0)

        zeros = crossings_nonzero_all(for_zeros.values)

        for i90, i270 in zip(phase_90, phase_270):
            mask = ((zeros > i90) & (zeros < i270)).nonzero()
            if len(mask[0]) == 1:
                phase_180.append(zeros[mask].item())
            else:
                phase_180.append(zeros[mask][-1].item())
                # phase_180.append(np.mean(zeros[mask]).astype(np.int))

        for i90, i270 in zip(phase_90[1:], phase_270):
            mask = ((zeros < i90) & (zeros > i270)).nonzero()
            if len(mask[0]) == 1:
                phase_0.append(zeros[mask].item())
            else:
                # print(zeros[mask])
                # dcpy.plots.linex(sig.time[[i90, np.mean(zeros[mask]).astype(np.int), i270]], color='k' )
                phase_0.append(zeros[mask][0].item())
                # phase_0.append(np.mean(zeros[mask]).astype(np.int))

                # choose mean value if multiple zero crossings

        # import IPython; IPython.core.debugger.set_trace()
        # zeros = sig.time[idx]

    elif algo_0_180 == "mean":
        for i90, i270 in zip(phase_90, phase_270):
            sig180 = (sig[i90] + sig[i270]) / 2
            phase_180.append(np.abs(sig[i90:i270] - sig180).argmin().values + i90)

        for i90, i270 in zip(phase_90[1:], phase_270):
            sig0 = (sig[i90] + sig[i270]) / 2
            phase_0.append(np.abs(sig[i270:i90] - sig0).argmin().values + i270)
    else:
        raise ValueError("algo_0_180 must be one of ['mean', 'zero-crossing']")

    longitude = sig.longitude.values.item()
    # print(sig.longitude)
    # if sig.longitude.values.item() == -110:
    #     print("fixing -110")
    #     #    import IPython; IPython.core.debugger.set_trace()
    #     print(phase_180[1])
    #     phase_180[1] = phase_180[1] + 70
    #     print(phase_180[1])

    if longitude == -110:
        print("fixing -110")
        phase_90[1] = phase_90[1] + 18
        phase_180[1] = phase_180[1] + 12

        phase_90[4] = phase_90[4] - 18
        phase_180[4] = phase_180[4] - 18
    elif longitude == -125:
        phase_90[3] = phase_90[3] - 24
        phase_180[3] = phase_180[3] - 6

    phase, period = merge_phase_label_period(
        sig, phase_0, phase_90, phase_180, phase_270, debug=debug,
    )

    if inserted_0:
        phase[: phase_270[0]] = np.nan
        period = period.where(~np.isnan(phase))

    # estimate ptp and filter out "weak" waves
    # this doesn't work so well
    tiw_ptp = calc_ptp(sig, period)
    # ptp_mask = tiw_ptp > tiw_ptp.quantile(dim="time", q=0.5)
    # phase = phase.where(ptp_mask)
    # period = period.where(ptp_mask)

    # Use SST gradient to refine
    lats = {
        -110: slice(0, 3),
        -125: slice(2, 5),
        -140: slice(0, 3),
        -155: slice(0, 3),
    }
    mean_grad = (
        grad.sel(latitude=lats[longitude])
        .mean("latitude")
        .assign_coords(tiw_phase=phase, period=period)
    )
    dt = mean_grad.time.diff("time").median("time")
    imax = (
        mean_grad.time.groupby(period).first()
        + mean_grad.groupby("period").apply(
            # make sure we don't get distracted by stuff near the beginning of the period
            lambda x: x.where(x.tiw_phase > 20).argmax("time")
        )
        * dt.values
    )
    bad_periods = imax.period[phase.sel(time=imax) < 90]
    print(phase.sel(time=imax))

    if len(bad_periods) > 0:
        # print(phase.sel(time=imax))
        warnings.warn(
            f"Found periods where SST front is before phase=90: {bad_periods.values}",
            UserWarning,
        )
        #  IPython; IPython.core.debugger.set_trace()

        if debug:
            plt.figure()
            mean_grad.plot(x="time")
            dcpy.plots.linex(imax)
            axphase = plt.gca().twinx()
            phase.plot(ax=axphase, color="k")

        gr = mean_grad.where(period.isin(bad_periods), drop=True)
        fixed = gr.groupby("period").map(fix_phase_using_sst_front, debug=debug)
        phase.loc[fixed.time] = fixed.values
        if debug:
            phase.plot(ax=axphase, color="r")

    return xr.merge([phase, period, tiw_ptp]).expand_dims("longitude")


def tiw_avg_filter_sst(sst, filt="bandpass", debug=False):

    if filt == "lowpass":
        sstfilt = xfilter.lowpass(
            sst.sel(latitude=slice(-1, 5)).mean("latitude"),
            coord="time",
            freq=1 / 15,
            cycles_per="D",
            num_discard=0,
        )
    elif filt == "bandpass":
        longitudes = sst.longitude.values

        kwargs = dict(coord="time", cycles_per="D", num_discard=0, method="pad",)

        lon_params = {
            -110: dict(freq=[1 / 40, 1 / 20], lats=slice(-1, 2)),
            -125: dict(freq=[1 / 60, 1 / 30], lats=slice(-1, 3)),
            -140: dict(freq=[1 / 60, 1 / 30], lats=slice(-1, 3)),
            -155: dict(freq=[1 / 60, 1 / 30], lats=slice(-1, 3)),
        }

        sstfilt = []
        for lon in longitudes:
            params = lon_params[lon]
            latmean = sst.sel(longitude=[lon], latitude=params["lats"]).mean("latitude")
            sstfilt.append(xfilter.bandpass(latmean, freq=params["freq"], **kwargs))

        sstfilt = xr.concat(sstfilt, dim="longitude").sortby(
            "longitude", ascending=False
        )

    return sstfilt


def get_tiw_phase_sst(sstfilt, gradsst, debug=False):

    ds = xr.Dataset()
    ds["sst"] = sstfilt
    ds["grad"] = gradsst
    output = ds.map_blocks(_find_phase_single_lon, kwargs={"debug": debug}).sortby(
        "longitude", ascending=False
    )

    return output["tiw_phase"], output["period"], output["tiw_ptp"]


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

        # ds["dens_euc"] = ds.dens.interp(
        #     depth=ds.eucmax, longitude=ds.longitude, method="linear"
        # )
        ds["dens_euc"] = euc.dens
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
            dKdt.groupby("time.day").plot(
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


def calc_ptp(sst, period=None, debug=False):
    """ Estimate TIW SST PTP amplitude. """

    def _calc_ptp(obj, dim="time"):
        obj = obj.unstack()
        obj -= obj.mean()
        return obj.max(dim) - obj.min(dim)

    if period is None:
        period = sst["period"]

    tiw_ptp = sst.groupby(period).map(_calc_ptp)

    if debug:
        plt.figure()
        tiw_ptp.plot(x="period", hue="longitude")

    tiw_ptp = (
        tiw_ptp.sel(period=period.dropna("time"))
        .reindex(time=period.time)
        .drop("period")
    )

    tiw_ptp.name = "tiw_ptp"
    tiw_ptp.attrs["description"] = "Peak to peak amplitude"

    return tiw_ptp


def estimate_shear_evolution_terms(ds):
    f = dcpy.oceans.coriolis(ds.latitude)
    uz = ds.u.differentiate("depth")
    vz = ds.v.differentiate("depth")

    b = -9.81 / 1035 * ds.dens
    #### zonal shear
    duzdt = xr.Dataset()
    # duzdt["shear"] = uz
    duzdt["xadv"] = -ds.u * ddx(uz)
    duzdt["yadv"] = -ds.v * ddy(uz)
    duzdt["str"] = uz * ddy(ds.v)
    duzdt["tilt"] = (f - ddy(ds.u)) * vz
    duzdt["vtilt"] = (f + ddx(ds.v) - ddy(ds.u)) * vz
    duzdt["htilt"] = -ddx(ds.v) * vz
    duzdt["buoy"] = -ddx(b)
    if "Um_Diss" in ds and "Um_Impl" in ds:
        duzdt["fric"] = (ds.Um_Diss + ds.Um_Impl).differentiate("depth")
    else:
        print("Skipping frictional term")
        duzdt["fric"] = xr.full_like(duzdt["vtilt"], np.nan)

    duzdt = duzdt.isel(longitude=1)
    duzdt.attrs["description"] = "Zonal shear evolution terms"

    duzdt["xadv"].attrs["term"] = "$-u ∂_xu_z$"
    duzdt["yadv"].attrs["term"] = "$-v ∂_yu_z$"
    duzdt["str"].attrs["term"] = "$u_z v_y$"
    duzdt["tilt"].attrs["term"] = "$(f-u_y) v_z$"
    duzdt["vtilt"].attrs["term"] = "$ζ v_z$"
    duzdt["htilt"].attrs["term"] = "$-v_x v_z$"
    duzdt["buoy"].attrs["term"] = "$-b_x$"
    duzdt["fric"].attrs["term"] = "$F^x_z$"

    #### meridional shear
    dvzdt = xr.Dataset()
    # dvzdt["shear"] = vz
    dvzdt["xadv"] = -ds.u * ddx(vz)
    dvzdt["yadv"] = -ds.v * ddy(vz)
    dvzdt["str"] = vz * ddx(ds.u)
    dvzdt["tilt"] = -(f + ddx(ds.v)) * uz
    dvzdt["vtilt"] = -(f + ddx(ds.v) - ddy(ds.u)) * uz
    dvzdt["htilt"] = -ddy(ds.u) * uz
    dvzdt["buoy"] = -ddy(b)

    if "Vm_Diss" in ds and "Vm_Impl" in ds:
        dvzdt["fric"] = (ds.Vm_Diss + ds.Vm_Impl).differentiate("depth")
    else:
        print("Skipping frictional term")
        dvzdt["fric"] = xr.full_like(duzdt["vtilt"], np.nan)

    dvzdt = dvzdt.isel(longitude=1)
    dvzdt.attrs["description"] = "Meridional shear evolution terms"

    dvzdt["xadv"].attrs["term"] = "$-u ∂_xv_z$"
    dvzdt["yadv"].attrs["term"] = "$-v ∂_yv_z$"
    dvzdt["str"].attrs["term"] = "$v_z u_x$"
    dvzdt["tilt"].attrs["term"] = "$-(f+v_x) u_z$"
    dvzdt["vtilt"].attrs["term"] = "$-ζ u_z$"
    dvzdt["htilt"].attrs["term"] = "$-u_y u_z$"
    dvzdt["buoy"].attrs["term"] = "$-b_y$"
    dvzdt["fric"].attrs["term"] = "$F^y_z$"

    for dset in [duzdt, dvzdt]:
        dset["xadv"].attrs["long_name"] = "zonal adv."
        dset["yadv"].attrs["long_name"] = "meridional adv."
        dset["str"].attrs["long_name"] = "stretching"
        dset["tilt"].attrs["long_name"] = "tilting"
        dset["vtilt"].attrs["long_name"] = "vvort tilting"
        dset["htilt"].attrs["long_name"] = "hvort tilting"
        dset["buoy"].attrs["long_name"] = "buoyancy"
        dset["fric"].attrs["long_name"] = "friction"

    return duzdt, dvzdt


def coare_fluxes_jra(ocean, forcing):
    """
    Calculates COARE3.5 bulk fluxes using MITgcm output and JRA forcing files.
    1. ignores rain for now.
    2. Remember to do forcing['time'] = forcing.time.dt.floor("h") to work around some
       weird bug.
    """

    import xcoare

    ocean = ocean.sel(depth=0, method="nearest", drop=True)

    sst = ocean.theta

    # mitgcm values
    ε = 0.97
    stefan = 5.67e-8
    albedo = 0.1

    coare = xcoare.xcoare35(
        u=np.hypot(forcing.uas, forcing.vas),
        zu=10,
        t=forcing.tas - 273.15,  # K to C
        zt=10,
        rh=None,
        zq=10,
        P=forcing.psl / 100,  # Pa to mbar
        ts=sst,
        Rs=forcing.rsds,
        Rl=forcing.rlds,
        lat=ocean.latitude,
        rain=forcing.prra / 1000 * 1000 / 3600,  # kg/m²/s to mm/hour
        jcool=False,
        qspec=forcing.huss,
        albedo=albedo,
        emissivity=ε,
    )

    flux = xr.Dataset()

    flux["long"] = -ε * ((stefan * (sst + 273.16) ** 4) - forcing.rlds)
    flux["short"] = (1 - albedo) * forcing.rsds  # using mitgcm albedo
    flux["sens"] = -coare.hsb
    flux["lat"] = -coare.hlb
    flux["netflux"] = flux.to_array().sum("variable")
    flux["stress"] = coare["tau"]

    wind_angle = np.arctan2(forcing.vas, forcing.uas)
    flux["taux"] = coare.tau * np.cos(wind_angle)
    flux["tauy"] = coare.tau * np.sin(wind_angle)
    return flux, coare


def calc_kpp_hbl(station, debug=False):

    station = station.sel(depth=slice(-500))
    h = xr.apply_ufunc(
        KPP.kpp,
        station.u,
        station.v,
        station.theta,
        station.salt,
        station.depth,
        np.arange(0, station.depth[-1] - 2.5, -2.5),
        station.taux / 1035,
        station.tauy / 1035,
        -(station.netflux - station.short) / 1035 / 3999,
        xr.zeros_like(station.netflux),
        -station.short / 1035 / 3999,
        kwargs=dict(
            COR=0, hbl=12, debug=False, r1=0.62, amu1=0.6, r2=1 - 0.62, amu2=20
        ),
        vectorize=True,
        # dask="allowed",  # too many assert statements to do this!
        input_core_dims=[["depth"]] * 5 + [["depth2"]] + [[],] * 5,
        # output_core_dims=[["time"]]
    )

    if debug:
        h.plot()
        # kpp.KPPhbl.plot()
        station.KPPhbl.plot()

    return h
