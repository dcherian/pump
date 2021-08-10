import dcpy.finestructure
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import xgcm


def _rename_to_theta_coordinates(means):
    """Rename from z_c, z_f to theta_c, theta_f"""

    means = means.copy(deep=True)
    z_c = means.z_c
    z_f = means.z_f

    means = means.drop_vars(["z_c", "z_f"]).set_coords(["theta_c", "theta_f"])
    means["theta_c"] = means.theta_c.squeeze()
    means["theta_f"] = means.theta_f.squeeze()
    means = means.swap_dims({"z_c": "theta_c", "z_f": "theta_f"}).assign_coords(
        {
            "z_c": ("theta_c", z_c.data),
            "z_f": ("theta_f", z_f.data),
        }
    )
    return means


def get_avg_dz(group):
    """Takes a binned group which is stacked in (depth, time)
    finds dz per profile (i.e. per timestep) and then averages in time
    """
    group = group.unstack()
    z = group.pres  # .where(group.chi.notnull())
    dz = z.max("depth") - z.min("depth")
    return dz.mean("time")


def regrid_chameleon(profiles, *, bins=None, debug=False, trim_mld=False):
    """Regrid chameleon profiles to temperature space.

    Also calculates wci.

    Parameters
    ----------

    profiles: xr.Dataset
        Actual data.
    bins: Sequence, optional
        Temperature bins.
    debug: bool, optional
        Make debugging plots
    trim_mld: bool, optional
        Use dcpy.finestructure.trim_mld_mode_water first?
    """

    trimmed = []
    for itime in range(profiles.sizes["time"]):
        profile = profiles.isel(time=itime)

        profile["theta"] = profile.theta.interpolate_na("depth")
        if profile.theta.count("depth") == 0:
            continue
        unsorted_depth = profile.depth
        profile = profile.sortby("theta", ascending=False)
        profile["depth"] = unsorted_depth

        profile = profile.sel(depth=profile.theta.notnull())

        if trim_mld:
            trimmed_ = dcpy.finestructure.trim_mld_mode_water(
                profile.drop("zeuc"), mode=False
            )
        else:
            trimmed_ = profile

        if trimmed_.sizes["depth"] > 0:
            trimmed.append(
                trimmed_.reset_coords().set_coords("time").expand_dims("time")
            )
    trimmed = xr.concat(trimmed, "time")

    if trim_mld:
        θmean = dcpy.finestructure.trim_mld_mode_water(
            profiles.mean("time").drop("zeuc"), mode=False
        ).theta
    else:
        θmean = profiles.theta.mean("time")
    if bins is None:
        # TODO: copy bin selection algorithm from mixingsoftware
        bins = np.sort(θmean.sel(depth=slice(20, 201, 10)))

    binned = trimmed[["eps", "chi", "u", "v", "pres", "theta", "salt"]].groupby_bins(
        "theta", bins=bins
    )

    means = binned.mean()
    means = means.set_coords("pres")

    means["nprof"] = (
        [],
        trimmed.sizes["time"],
        {"description": "number of profiles in average"},
    )
    if trim_mld:
        means["Tmld"] = ([], trimmed.Tmld.mean(), {"description": "mean Tmld"})
        means["σmld"] = ([], trimmed.σmld.mean(), {"description": "mean σmld"})

    # TODO: this could be better. I could interp a really smooth profile and then get distances out.
    # This loses heat I think; could do some conservative regridding
    means.coords["dz"] = binned.map(get_avg_dz)

    means = means.rename_dims({"theta_bins": "z_c"})

    # zbot = trimmed.depth[-1].data
    zbot = 201
    zbot = np.abs(θmean - bins.min()).idxmin("depth").data
    means.coords["z_f"] = (
        "z_f",
        np.insert(zbot - np.cumsum(means.dz.data), 0, zbot),
    )
    means.coords["z_c"] = (
        "z_c",
        (means.z_f.data[:-1] + means.z_f.data[1:]) / 2,
    )

    ρcp = 1025 * 4000
    means["theta_f"] = ("z_f", bins)
    means["dT"] = ("z_c", np.diff(bins))

    means["dTdz"] = means["dT"] / means["dz"]
    means.coords["nobs"] = binned.count().chi

    means["Kt"] = means.chi / 2 / means["dTdz"] ** 2
    means.Kt.attrs["long_name"] = "$K_T$"
    means.Kt.attrs["units"] = "m²/s"

    means["Jq"] = -1 * ρcp * means.chi / 2 / means["dTdz"]
    means.Jq.attrs["long_name"] = "$J_q$"
    means.Jq.attrs["units"] = "W/m²"

    grid = xgcm.Grid(
        means,
        coords={"Z": {"outer": "z_f", "center": "z_c"}},
        metrics={("Z",): ("dz")},
        boundary="fill",
        fill_value=np.nan,
    )

    means["dJdz"] = grid.diff(means.Jq, "Z") / grid.diff(means.z_c, "Z")
    means["theta_c"] = grid.interp(means.theta_f, axis="Z")
    means["dJdT"] = -grid.diff(means.Jq, "Z") / grid.diff(means.theta_c, "Z")
    means["wci"] = means.dJdT / ρcp
    means.wci.attrs["long_name"] = "$w_{ci}$"
    means.wci.attrs["units"] = "m/s"

    means = _rename_to_theta_coordinates(means)

    if debug:
        # f = plt.figure(constrained_layout=True)
        # f.add_grispec
        f, ax = plt.subplots(1, 3, sharey=True, constrained_layout=True)

        profiles.theta.mean("time").cf.plot(ax=ax[0])
        trimmed.theta.mean("time").cf.plot(ax=ax[0])
        means.theta_f.plot(ax=ax[0], y="z_f", marker="x")

        # trimmed.theta.median("time").cf.plot(ax=ax[0])
        dcpy.plots.linex(bins, ax=ax[0])

        means.Jq.cf.plot.step(ax=ax[1], y="z_c", where="mid")

        axdj = ax[1].twiny()
        means.dJdz.cf.plot.step(ax=axdj, y="z_f", color="C3", where="mid")
        dcpy.plots.set_axes_color(axdj, "C3", spine="top")
        dcpy.plots.set_axes_color(ax[1], "C0", spine="bottom")

        means.wci.cf.plot.step(ax=ax[2], y="z_f", xlim=(-1e-4, 1e-4))

        dcpy.plots.linex(0, ax=ax[2])
        dcpy.plots.linex(0, ax=axdj, color="r")
        dcpy.plots.clean_axes(ax)

        plt.figure()
        trimmed.chi.cf.plot(
            x="time",
            robust=True,
            norm=mpl.colors.LogNorm(),
            cmap=mpl.cm.Spectral_r,
            cbar_kwargs={"orientation": "horizontal"},
        )
        trimmed.theta.cf.plot.contour(
            x="time",
            levels=bins,
            linewidths=1,
            colors="k",  # ylim=(120, 0)
        )
        plt.tight_layout()

    coord_names = means.coords.keys()
    means = (
        means.reset_coords()
        .expand_dims(time=[profiles.time[0].dt.floor("H").data])
        .set_coords(["nprof"])
        .set_coords(coord_names)
    )
    if trim_mld:
        means = means.set_coords(["Tmld", "σmld"])
    return means
