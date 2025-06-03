# %% import libraries
import xarray as xr
import pandas as pd
from aws_ir import aws_loading, cpcir_loading, aws_remapping
from aws_ir.combine import align_cpcir, package_ml_xy


# %%
def save_ml_dataset(time_start, time_end):

    timerange = slice(time_start, time_end)

    # %% Load AWS data
    files_aws = aws_loading.get_files_l1b(
        timerange=timerange,
    )
    if len(files_aws) == 0:
        print("There is no AWS data in the time range, skip.")
        return

    # load AWS data
    ds_aws = aws_loading.load_multiple_files_l1b(files_aws, apply_fixes=True)
    ds_aws = ds_aws.sel(time=timerange, n_fovs=slice(40, 100))
    ds_aws = ds_aws.where((ds_aws.flag_bad_data == 0).compute(), drop=True)

    if ds_aws.n_scans.size == 0:
        print("There is no scan met the condition, skip.")
        return
    # %% remap AWS data
    # skip AWS1X remapping and join the other groups later
    ds_aws_group1 = ds_aws.sel(
        n_channels=[
            "AWS11",
            "AWS12",
            "AWS13",
            "AWS14",
            "AWS15",
            "AWS16",
            "AWS17",
            "AWS18",
        ]
    ).drop_vars(["aws_lat", "aws_lon"])

    # remap AWS2X and AwS4X to AWS3X
    ds_aws_remapped_group234 = aws_remapping.remap_interp(
        ds_aws.sel(
            n_channels=[
                "AWS21",
                "AWS31",
                "AWS32",
                "AWS33",
                "AWS34",
                "AWS35",
                "AWS36",
                "AWS41",
                "AWS42",
                "AWS43",
                "AWS44",
            ],
        ),
        remap_to_ch="AWS33",
        method="nearest",
        fill_distance=True,
    )

    # join all channels
    ds_aws_remapped = xr.merge([ds_aws_group1, ds_aws_remapped_group234])

    # remove points outside the CPCIR latitude coverage
    ds_aws_remapped = ds_aws_remapped.where(
        (ds_aws_remapped.aws_lat.pipe(abs) < 60).compute(),
        drop=True,
    )

    if ds_aws_remapped.n_scans.size == 0:
        print("There is no scans within +-60 degrees in latitude")
        return
    # %% Load CPCIR data
    files_cpcir = cpcir_loading.get_cpcir_fileset(
        timerange=slice(ds_aws.time.min(), ds_aws.time.max()),
    )
    ds_cpcir = cpcir_loading.get_cpcir_ds(files_cpcir)

    # %% Align CPCIR to AWS data
    ds_cpcir_aligned = align_cpcir(ds_cpcir, ds_aws_remapped)

    # compute data
    Xy = package_ml_xy(ds_aws_remapped, ds_cpcir_aligned)

    # %% save data
    Xy.reset_index("n_samples").to_netcdf(
        f"./data/aws_cpcir_{time_start.replace('-', '').replace(':', '').replace('T', '')}_{time_end.replace('-', '').replace(':', '').replace('T', '')}.nc"
    )


# %%
time_start = "2025-05-01T00:00:00"
time_end = "2025-05-31T23:00:00"
date_range = pd.date_range(time_start, end=time_end, freq="1h")
for start in date_range:
    print(start)
    end = start + pd.Timedelta("1h")
    start = start.strftime("%Y-%m-%dT%X")
    end = end.strftime("%Y-%m-%dT%X")
    save_ml_dataset(start, end)
