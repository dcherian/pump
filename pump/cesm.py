import xarray as xr


def read_cesm(dirname):
    kwargs = dict(
        data_vars="minimal",
        coords="minimal",
        compat="override",
        concat_dim="time",
        combine="by_coords",
        parallel=True,
        chunks={"z_t": 7, "nlat": 600, "nlon": 1200},
    )

    def preprocess(ds):
        if "transport_components" in ds:
            return ds.drop(["transport_components", "transport_regions"])
        else:
            return ds

    if "CESM-LE" in dirname:
        kwargs["preprocess"] = preprocess
        u = xr.open_mfdataset(dirname + "/*UVEL/*", **kwargs)
        T = xr.open_mfdataset(dirname + "/*TEMP/*", **kwargs)
        S = xr.open_mfdataset(dirname + "/*SALT/*", **kwargs)

    else:
        u = xr.open_mfdataset(dirname + "/*UVEL.*", **kwargs)
        T = xr.open_mfdataset(dirname + "/*TEMP.*", **kwargs)
        S = xr.open_mfdataset(dirname + "/*SALT.*", **kwargs)

    cesm = xr.merge([u, T, S], compat="override").rename(
        {"UVEL": "u", "TEMP": "temp", "SALT": "salt"}
    )

    return cesm

# pop-tools says that this is using mdjwf
    # cesm['dens'] = pump.mdjwf.dens(cesm.salt, cesm.temp, cesm.z_t)

    # cesm = cesm.roll(nlon=-300, roll_coords=True)
    # cesm = cesm.isel(nlat=slice(1120, 1260), nlon=slice(2000, 3600))
    # cesm["ULONG"] = xr.where(cesm["ULONG"] < 0, cesm["ULONG"] + 360, cesm["ULONG"])
    # cesm["ULONG"] -= 360
    # cesm["TLONG"] -= 360

    # #cesm["longitude"] = (("longitude"), cesm["ULONG"].values)
    # cesm = cesm.rename({"z_t": "depth"})
    # cesm["depth"] /= -100
    # cesm["depth"].attrs["units"] = "m"
    # cesm = cesm.sel(depth=slice(0, -600))
    # cesm["u"] /= 100
    # cesm["u"].attrs["units"] = "m/s"

    # return cesm


def read_small():
    return read_cesm(
        "/glade/p/cesm/community/ASD-HIGH-RES-CESM1/hybrid_v5_rel04_BC5_ne120_t12_pop62/"
        "ocn/proc/tseries/monthly"
    )


def read_cesm_le():
    return read_cesm(
        "/glade/p/cesm/community/CESM-LE/data/CESM-CAM5-BGC-LE/"
        "ocn/proc/tseries/monthly"
    )
