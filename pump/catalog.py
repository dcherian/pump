import ast
from pathlib import Path

import intake

# MOM6 run catalog
# This is a dictionary mapping a "short name" to a tuple("descriptive name for legend", "case name")
# This information is embedded in the catalog.
catalog_dict = {
    "baseline.001": (
        "baseline",
        "gmom.e23.GJRAv3.TL319_t061_zstar_N65.baseline.001.mixpods",
    ),
    "epbl": ("ePBL", "gmom.e23.GJRAv3.TL319_t061_zstar_N65.baseline.epbl.001.mixpods"),
    "kpp.lmd.002": (
        "KPP Ri0=0.5",
        "gmom.e23.GJRAv3.TL319_t061_zstar_N65.baseline.kpp.lmd.002.mixpods",
    ),
    "kpp.lmd.003": (
        "KPP Ri0=0.5, Ric=0.2,",
        "gmom.e23.GJRAv3.TL319_t061_zstar_N65.baseline.kpp.lmd.003.mixpods",
    ),
    "kpp.lmd.004": (
        "KPP ν0=2.5, Ric=0.2, Ri0=0.5",
        "gmom.e23.GJRAv3.TL319_t061_zstar_N65.baseline.kpp.lmd.004.mixpods",
    ),
    "baseline.N150": (
        "baseline N=150",
        "gmom.e23.GJRAv3.TL319_t061_zstar_N150.baseline.mixpods",
    ),
    "kpp.lmd.004.N150": (
        "KPP ν0=2.5, Ric=0.2, Ri0=0.5, N=150",
        "gmom.e23.GJRAv3.TL319_t061_zstar_N150.kpp.lmd.004.mixpods",
    ),
    "baseline.hb": (
        "baseline",
        "gmom.e23.GJRAv3.TL319_t061_zstar_N65.baseline.hb",
    ),
    "new_baseline.hb": (
        "KD=0, KV=0",
        "gmom.e23.GJRAv3.TL319_t061_zstar_N65.new_baseline.hb",
    ),
    "new_baseline.kpp.lmd.004": (
        "KPP ν0=2.5, Ric=0.2, Ri0=0.5",
        "gmom.e23.GJRAv3.TL319_t061_zstar_N65.new_baseline.kpp.lmd.004.mixpods",
    ),
    "new_baseline.kpp.lmd.005": (
        "KPP ν0=2.5, Ri0=0.5",
        "gmom.e23.GJRAv3.TL319_t061_zstar_N65.new_baseline.kpp.lmd.005.mixpods",
    ),
}


DESCRIPTIONS = {
    casename: description for shortname, (description, casename) in catalog_dict.items()
}


METRIC_VARS = {
    "Coriolis",
    "areacello",
    "areacello_bu",
    "areacello_cu",
    "areacello_cv",
    "cos_rot",
    "deptho",
    "geolat",
    "geolat_c",
    "geolat_u",
    "geolat_v",
    "geolon",
    "geolon_c",
    "geolon_u",
    "geolon_v",
    "nv",
    "sin_rot",
    "wet",
    "wet_c",
    "wet_u",
    "wet_v",
    "xh",
    "xq",
    "yh",
    "yq",
    "zi",
}

IGNORE_VARS = {
    "average_DT",
    "average_T1",
    "average_T2",
    "time",
    "time_bnds",
    "z_i",
    "z_l",
}


def add_description(info):
    info["description"] = DESCRIPTIONS[info["casename"]]
    return info


def make_short_name(info):
    casename = info["casename"]
    casename_split = casename.split(".")
    trimmed = casename_split[4:]
    if trimmed[-1] == "mixpods":
        trimmed = trimmed[:-1]
    info["shortname"] = ".".join(trimmed)
    return info


def parse_cesm_mom6_kerchunk_json(file, modifiers=None, storage_options=None):
    """Parser for CESM MOM6 reference JSON files"""
    import itertools
    import warnings
    from pathlib import Path

    import datatree
    import fsspec

    import xarray as xr

    if storage_options is None:
        storage_options = {}

    path = Path(file)

    info = dict()
    info["casename"] = path.parts[-4]
    info["stream"] = path.stem
    info["path"] = file
    info["baseline"] = "new" if "new_baseline" in info["casename"] else "old"
    info["levels"] = int(path.parts[-4].split(".")[3].split("_")[-1][1:])

    fs = fsspec.filesystem("reference", fo=file, **storage_options)
    mapper = fs.get_mapper("")

    if path.stem == "combined":
        info["frequency"] = "N/A"
        dt = datatree.open_datatree(
            mapper, engine="zarr", use_cftime=True, consolidated=False
        )
        variables = itertools.chain(*[node.variables.keys() for node in dt.subtree])
        if not dt:
            warnings.warn(f"bad file: {file}", RuntimeWarning)
            dt.close()
            return
    else:
        ds = xr.open_zarr(
            mapper,
            use_cftime=True,
            consolidated=False,
            chunks={},
        )
        if not ds:
            warnings.warn(f"bad file: {file}", RuntimeWarning)
            ds.close()
            return
        # TODO: Use keith's util
        info["frequency"] = "daily" if xr.infer_freq(ds.time) == "D" else "monthly"
        variables = ds.variables.keys()

    info["variables"] = sorted(set(variables) - set(METRIC_VARS) - set(IGNORE_VARS))

    if modifiers is not None:
        if callable(modifiers):
            modifiers = (modifiers,)
        for modifier in modifiers:
            info = modifier(info)

    return info


def build_mom6_catalog():
    from ecgtools import Builder

    from .mixpods import ROOT

    ROOT = Path(ROOT)

    builder = Builder(paths=tuple(str(p) for p in ROOT.glob("**/run/jsons/")), depth=0)

    from functools import partial

    builder.build(
        parsing_func=partial(
            parse_cesm_mom6_kerchunk_json, modifiers=(make_short_name, add_description)
        )
    )

    builder.save(
        name="pump-mom6-catalog",
        # Column name including filepath
        path_column_name="path",
        # Column name including variables
        variable_column_name="variables",
        # Data file format - could be netcdf or zarr or reference (in this case, netcdf)
        data_format="reference",
        # Which attributes to groupby when reading in variables using intake-esm
        groupby_attrs=["shortname", "stream"],
        # Aggregations which are fed into xarray when reading in data using intake
        aggregations=[],
        # directory to save catalog to
        directory=f"{ROOT}/catalogs/",
        # these are reference catalogs, so embed in JSON
        catalog_type="dict",
    )

    open_mom6_catalog()


# path = (
#    "/glade/campaign/cgd/oce/projects/pump/cesm/gmom.e23.GJRAv3.TL319_t061_zstar_N65.new_baseline.hb/"
#    "run/jsons/combined.json"
# )
# parse_cesm_mom6_kerchunk_json(str(path), modifiers=(make_short_name, add_description))


def open_mom6_catalog():
    from .mixpods import ROOT

    return intake.open_esm_datastore(
        f"{ROOT}/catalogs/pump-mom6-catalog.json",
        read_csv_kwargs={"converters": {"variables": ast.literal_eval}},
    )


def catalog_to_grid(data_catalog):
    import ipyaggrid

    df = data_catalog.df

    column_defs = [
        {
            "headerName": "shortname",
            "field": "shortname",
            "rowGroup": False,
            "pinned": True,
        },
        {"headerName": "stream", "field": "stream", "rowGroup": False},
        {"headerName": "baseline", "field": "baseline"},
        {"headerName": "frequency", "field": "frequency", "rowGroup": False},
        {"headerName": "levels", "field": "levels", "rowGroup": False},
        {"headerName": "variables", "field": "variables", "autoHeight": True},
        {"headerName": "casename", "field": "casename", "rowGroup": False},
        {"headerName": "path", "field": "path", "rowGroup": False},
    ]

    grid_options = {
        "columnDefs": column_defs,
        "defaultColDef": {
            "resizable": True,
            "editable": False,
            "filter": True,
            "sortable": True,
        },
        "colResizeDefault": True,
        "enableRangeSelection": True,
        "statusBar": {  # new syntax since 19.0
            "statusPanels": [
                {"statusPanel": "agTotalRowCountComponent", "align": "left"},
                {"statusPanel": "agFilteredRowCountComponent"},
                {"statusPanel": "agSelectedRowCountComponent"},
                {"statusPanel": "agAggregationComponent"},
            ]
        },
        # "enableRangeHandle": True,
    }

    g = ipyaggrid.Grid(
        grid_data=df,
        grid_options=grid_options,
        quick_filter=True,
        export_csv=False,
        export_excel=False,
        export_mode="buttons",
        export_to_df=True,
        theme="ag-theme-balham",
        # show_toggle_edit=False,
        # show_toggle_delete=False,
        columns_fit="auto",
        # index=False,
        # keep_multiindex=False,
    )
    return g


def interact():
    cat = open_mom6_catalog()
    return catalog_to_grid(cat)
