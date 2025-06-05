import datetime
import xarray as xr


def align_cpcir(ds_cpcir, ds_aws_remapped):
    """
    Remap the CPCIR data to the AWS grid.
    """
    # Select the nearest time of AWS data
    ds_cpcir_aligned = ds_cpcir.sel(
        time=ds_aws_remapped.time,
        method="nearest",
        tolerance=datetime.timedelta(minutes=30),
    )
    # linear spatial interpolation of the CPCIR data to the AWS grid
    # TODO: copy the the last longitude grid to the first one
    ds_cpcir_aligned = ds_cpcir_aligned.interp(
        lat=ds_aws_remapped.aws_lat,
        lon=ds_aws_remapped.aws_lon,
        method="linear",
        # kwargs={"fill_value": "extrapolate"},
    )
    return ds_cpcir_aligned.set_xindex("time")


def package_ml_xy(ds_aws_remapped, ds_cpcir_aligned):
    """
    Package the AWS and CPCIR data into a 2D array for X and 1D array for y.
    """
    # Select the variables to be used for X and y
    x_vars = "aws_toa_brightness_temperature"
    y_vars = "Tb"
    # Create the X and y arrays
    X = (
        ds_aws_remapped[x_vars]
        # .to_array()
        .stack(
            dict(
                n_samples=["n_scans", "n_fovs"],
                # features=["n_channels", "variable"],
            )
        )
    ).compute()
    y = (
        (
            ds_cpcir_aligned[y_vars]
            # .to_array()
            .stack(
                dict(
                    n_samples=["n_scans", "n_fovs"],
                    # features=["variable"],
                )
            )
        )
        # replace dims coords to be able to merge
        .drop_vars(["n_samples", "time", "n_fovs"])
        .assign_coords(n_samples=X["n_samples"])
        .compute()
    )

    mask = y.notnull().squeeze()
    Xy = xr.merge([X, y]).where(mask, drop=True)
    return Xy
