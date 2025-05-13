import datetime


def align_cpcir(ds_cpcir, ds_aws_remapped):
    """
    Remap the CPCIR data to the AWS grid.
    """
    # Select the nearest time of AWS data
    ds_cpcir_remapped = ds_cpcir.sel(
        time=ds_aws_remapped.time,
        method="nearest",
        tolerance=datetime.timedelta(minutes=30),
    )
    # linear interpolation of the CPCIR data to the AWS grid
    ds_cpcir_remapped = ds_cpcir_remapped.interp(
        lat=ds_aws_remapped.aws_lat,
        lon=ds_aws_remapped.aws_lon,
        method="linear",
        # kwargs={"fill_value": "extrapolate"},
    )
    return ds_cpcir_remapped.set_xindex("time")


def package_ml_xy(ds_aws_remapped, ds_cpcir_aligned):
    """
    Package the AWS and CPCIR data into a 2D array for X and 1D array for y.
    """
    # Select the variables to be used for X and y
    x_vars = ["aws_toa_brightness_temperature"]
    y_vars = ["Tb"]
    # Create the X and y arrays
    X = (
        ds_aws_remapped[x_vars]
        .to_array()
        .stack(
            dict(
                n_samples=["n_scans", "n_fovs"],
                features=["n_channels", "variable"],
            )
        )
    ).compute()
    y = (
        ds_cpcir_aligned[y_vars]
        .to_array()
        .stack(
            dict(
                n_samples=["n_scans", "n_fovs"],
                features=["variable"],
            )
        )
    ).compute()

    # TODO: preserve dims and coords in output X, y
    # mask = y.notnull().squeeze()
    # y = y.where(mask, drop=True)

    # mask.drop_vars(['n_samples', 'time', 'n_fovs'])['n_samples'] = X.n_samples
    # X = X.where(mask, drop=True)

    mask = y.notnull().squeeze()
    X = X.values[mask, :]
    y = y.values[mask]
    return X, y
