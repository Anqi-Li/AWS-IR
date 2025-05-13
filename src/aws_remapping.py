from datetime import datetime

import dask
import dask.diagnostics
import numpy as np
import scipy
import xarray as xr

from .aws_loading import AWSChannel
from .utils import get_git_revision

DSCAN = 8


# %% remapping
def remap_interp(
    ds, remap_to_ch: str, method: str = "linear", fill_distance=False
) -> xr.Dataset:
    """
    Basic remap channels to lat/lon grid specified by remap_to_ch

    The remapping done as an interpolation (linear is default). All implemented
    in a basic fashion and code not very fast.

    NOTE: This performs a deep copy of the input dataset.

    :param ds:     Dataset with channels
    :param remap_to_ch: channel to remap data to.
    :param method:  Interpolation method. Select "nearest" for faster remapping,
                    but less accurate.
    :param fill_distance: Create new fields in returned dataset with distance to remap target.

    :return: Copied input dataset remapped Ta values and lat/lon values for channels.
    """
    remap_to_ch = AWSChannel.from_name(remap_to_ch)
    assert (
        remap_to_ch.value.name in ds.n_channels.values
    ), "Target channel to remap to needs to be in the provided dataset."

    # Ignore scans with bad time/lat/lon

    channels = [AWSChannel.from_name(ch_name) for ch_name in ds.n_channels.values]
    channels_to_remap = [
        ch
        for ch in channels
        if ch.value.index_geo_group != remap_to_ch.value.index_geo_group
    ]

    ichannels_to_remap = [
        i
        for i, ch_name in enumerate(ds.n_channels.values)
        if AWSChannel.from_name(ch_name) in channels_to_remap
    ]

    nscan = ds.n_scans.size

    ds_remapped = ds.copy(deep=True)

    # Split scans into chunks and create delayed dask jobs to perform the remapping in parallel
    chunk_size = 64
    results_ta = []
    results_distance = []
    for iscan_chunk_start in range(0, nscan, chunk_size):
        # Range to perform interpolation for
        scan_range = slice(
            iscan_chunk_start, min(nscan, iscan_chunk_start + chunk_size)
        )

        # Range with margin to get extra data for interpolation
        scan_range_with_margin = slice(
            max(0, scan_range.start - DSCAN), min(nscan, scan_range.stop + DSCAN)
        )

        latlons_for_chunk = (
            ds["aws_lat"][dict(n_scans=scan_range_with_margin)],
            ds["aws_lon"][dict(n_scans=scan_range_with_margin)],
        )
        values_for_chunk = ds["aws_toa_brightness_temperature"][
            dict(n_scans=scan_range_with_margin)
        ]

        result_delayed_ta, result_delayed_distance = remap_scan_chunk(
            remap_to_ch,
            channels_to_remap,
            latlons_for_chunk,
            values_for_chunk,
            scan_range,
            scan_range_with_margin,
            fill_distance=fill_distance,
            dscan=DSCAN,
            method=method,
        )
        results_ta.append(result_delayed_ta)
        if fill_distance:
            results_distance.append(result_delayed_distance)

    # Compute the results
    with dask.diagnostics.ProgressBar():
        results = dask.compute({"ta": results_ta, "distance": results_distance})[0]

    # Concatenate the computed chunks and fill the data...
    concatenated_result_ta = np.concatenate(results["ta"], axis=0)
    ds_remapped["aws_toa_brightness_temperature"][
        dict(n_channels=ichannels_to_remap)
    ] = concatenated_result_ta

    if fill_distance:
        concatenated_result_distance = np.concatenate(results["distance"], axis=0)
        ds_remapped["remap_distance"] = xr.DataArray(
            data=concatenated_result_distance.astype(np.float32),
            dims=["n_scans", "n_fovs", "n_geo_groups"],
            attrs=dict(
                description="Distance between original footprint and remapped footrpint for each channel.",
                unit="m",
            ),
        )

    # Update lat/lon data for remapped channels.
    # TODO: Set as coordinate?
    ds_remapped["aws_lat"] = ds["aws_lat"].sel(
        n_geo_groups=remap_to_ch.value.geo_group_name
    )
    ds_remapped["aws_lon"] = ds["aws_lon"].sel(
        n_geo_groups=remap_to_ch.value.geo_group_name
    )

    ds_remapped.attrs["history"] = (
        ds_remapped.attrs.get("history", "")
        + f"{datetime.now()} (aws_processing: {get_git_revision()}): Remapped `aws_toa_brightness_temperature` values to lat/lon of channel {remap_to_ch.value.name}.\n"
    )

    return ds_remapped


@dask.delayed(nout=2)
def remap_scan_chunk(
    channel_target,
    channels_to_remap,
    latlons_for_chunk,
    values_for_chunk,
    scan_range,
    scan_range_with_margin,
    fill_distance,
    dscan,
    method,
):
    """
    Perform remapping on a chunk of data

    Decorated as a @dask.delayed function to allow for parallel computation

    :param latlons_for_chunk: tuple with (lat[n_scans, n_fovs, n_chans], lon[n_scans, n_fovs, n_chans])
    :param values_for_chunk: values to remap with shape: [n_scans, n_fovs, n_chans]
    :param channels: AWSChannel items corresponding to the n_chans dimension of latlons and values.
    :param channels_to_remap: The subset of `channels` with channels that should be remapped.
    :param scan_range: Range of scan indices in original dataset to perform remapping for
    :param scan_range_margin: Range of scan indices in original dataset of the given chunk
    :param fill_distance: Weather or not to calculate the remapping distance

    :return: Returns tuple with remapped values and remap distances ([n_scans, n_fovs, n_chans])
    """
    for i in [0, 1]:
        assert (
            latlons_for_chunk[i].sizes["n_scans"] == values_for_chunk.sizes["n_scans"]
        )
        assert latlons_for_chunk[i].sizes["n_fovs"] == values_for_chunk.sizes["n_fovs"]

    assert (
        scan_range.start >= scan_range_with_margin.start
        and scan_range.stop <= scan_range_with_margin.stop
    ), "scan_range should be a sub range of scan_range_margin"

    nfovs = values_for_chunk.sizes["n_fovs"]
    nscan = scan_range.stop - scan_range.start

    # Allocate chunk outputs
    # TODO: edit in place instead
    out_values = np.empty((nscan, nfovs, len(channels_to_remap)))
    out_values.fill(np.nan)

    out_group_distances = None
    if fill_distance:
        out_group_distances = np.empty((nscan, nfovs, 4))
        out_group_distances.fill(np.nan)

    scan_range_offset = scan_range.start - scan_range_with_margin.start

    # If close to longitude 360, shift to [-180, 180]
    if latlons_for_chunk[1].max() > 355:
        latlons_for_chunk[1].loc[:, :] = shift_lons(latlons_for_chunk[1].values)

    for i, _ in enumerate(range(nscan)):
        iscan_in_chunk = scan_range_offset + i
        # Consider changing remap_scan to modify in place.
        out_values_scan, out_distances_scan = remap_scan(
            channel_target,
            channels_to_remap,
            iscan_in_chunk,
            latlons_for_chunk,
            values_for_chunk,
            fill_distance=fill_distance,
            dscan=dscan,
            method=method,
        )
        out_values[i, :, :] = out_values_scan
        if fill_distance:
            out_group_distances[i, :, :] = out_distances_scan

    return out_values, out_group_distances


def remap_scan(
    channel_target,
    channels_to_remap,
    iscan_in_chunk,
    latlons_for_chunk,
    values_for_chunk,
    fill_distance,
    dscan,
    method,
):
    "Return the interpolated data [iscan, :, i_channels_to_remap]"
    for i in [0, 1]:
        assert (
            latlons_for_chunk[i].sizes["n_scans"] == values_for_chunk.sizes["n_scans"]
        )
        assert latlons_for_chunk[i].sizes["n_fovs"] == values_for_chunk.sizes["n_fovs"]

    nfovs = values_for_chunk.shape[1]
    nscans_in_chunk = values_for_chunk.shape[0]

    # Allocate output
    # TODO: better to not allocate, and edit in place from some input...
    out = np.empty((nfovs, len(channels_to_remap)))
    out.fill(np.nan)

    out_distance = None
    if fill_distance:
        out_distance = np.empty((nfovs, 4))
        out_distance.fill(np.nan)
        out_distance[:, channel_target.value.index_geo_group] = 0.0

    target_geo_group = channel_target.value.geo_group_name

    points_target = (
        latlons_for_chunk[0]
        .loc[dict(n_geo_groups=target_geo_group)]
        .isel(n_scans=iscan_in_chunk),
        latlons_for_chunk[1]
        .loc[dict(n_geo_groups=target_geo_group)]
        .isel(n_scans=iscan_in_chunk),
    )
    if np.isnan(points_target[0]).any() or np.isnan(points_target[1]).any():
        # Skip remapping for this scan.
        return out, out_distance

    scan_range_source = slice(
        max(0, iscan_in_chunk - dscan), min(nscans_in_chunk, iscan_in_chunk + dscan + 1)
    )

    groups_that_have_had_distance_calculated = []
    for i, ch in enumerate(channels_to_remap):
        points_source = (
            latlons_for_chunk[0]
            .loc[dict(n_geo_groups=ch.value.geo_group_name)]
            .isel(n_scans=scan_range_source)
            .values.flatten(),
            latlons_for_chunk[1]
            .loc[dict(n_geo_groups=ch.value.geo_group_name)]
            .isel(n_scans=scan_range_source)
            .values.flatten(),
        )
        mask_valid = ~np.isnan(points_source[0]) & ~np.isnan(points_source[1])

        points_source = (
            points_source[0][mask_valid],
            points_source[1][mask_valid],
        )

        out[:, i] = scipy.interpolate.griddata(
            points_source,
            values_for_chunk.loc[
                dict(n_scans=scan_range_source, n_channels=ch.value.name)
            ].values.flatten()[mask_valid],
            points_target,
            method=method,
        )

        current_channel_group_index = ch.value.index_geo_group
        if (
            fill_distance
            and current_channel_group_index
            not in groups_that_have_had_distance_calculated
        ):
            idnum = np.arange(np.size(points_source[0]))
            ids = scipy.interpolate.griddata(
                points_source, idnum, points_target, method="nearest"
            )
            ii = ids[:].astype(int)
            distance = sphdist(
                points_target[0][:],
                points_target[1][:],
                points_source[0][ii],
                points_source[1][ii],
            )
            out_distance[:, current_channel_group_index] = distance
            groups_that_have_had_distance_calculated.append(current_channel_group_index)

    return out, out_distance


# %% geometric functions
def sphdist(
    lat0: np.ndarray,
    lon0: np.ndarray,
    lat1: np.ndarray,
    lon1: np.ndarray,
) -> np.ndarray:
    """
    Simple calculation of "as-the-crow-flies" distance between positions (lat0, lon0) and (lat1, lon1).

    "return": Distance.
    """
    mlat = np.deg2rad(lat0)
    mlon = np.deg2rad(lon0)
    plat = np.deg2rad(lat1)
    plon = np.deg2rad(lon1)
    v = np.array(
        np.sin(mlat) * np.sin(plat) + np.cos(mlat) * np.cos(plat) * np.cos(mlon - plon)
    )

    eps = 1e-8
    assert (
        (v <= 1.0 + eps) & (v >= -1.0 - eps)
    ).all(), "Clipping values, not a valid assumption"
    v_clipped = np.clip(v, a_min=-1.0, a_max=1.0, out=v)

    return 6371.01e3 * np.arccos(v_clipped, out=v)


def shift_lons(lons: np.ndarray) -> np.ndarray:
    """
    Shift longitudes to [-180, 180], from [0, 360].
    """
    return np.where(lons <= 180, lons, lons - 360)
