import os
import socket
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Iterable

import dask.array as da
import numpy as np
import pandas as pd
import parse
import xarray as xr
from tqdm import tqdm

from .utils import timeslice_cast

FILEPATTERN_L1B = "W_NO-KSAT-Tromso,SAT,{platform_name}-MWR-1B-RAD_C_OHB__{processing_time:%Y%m%d%H%M%S}_G_O_{start_time:%Y%m%d%H%M%S}_{end_time:%Y%m%d%H%M%S}_C_N____.nc"

DATADIR_L1B = os.environ.get(
    "AWSPROCESSING_DATADIR",
    (
        "/data/s6/L1/AWS/L1B"
        if socket.gethostname() == "geo-c"
        else Path(__file__).parent.parent.parent / "data"
    ),
)


@dataclass
class AWSChannelID:
    name: str
    geo_group_name: str
    index0: int
    index_geo_group: int


class AWSChannel(Enum):
    AWS11 = AWSChannelID("AWS11", "AWS1X", 0, 0)
    AWS12 = AWSChannelID("AWS12", "AWS1X", 1, 0)
    AWS13 = AWSChannelID("AWS13", "AWS1X", 2, 0)
    AWS14 = AWSChannelID("AWS14", "AWS1X", 3, 0)
    AWS15 = AWSChannelID("AWS15", "AWS1X", 4, 0)
    AWS16 = AWSChannelID("AWS16", "AWS1X", 5, 0)
    AWS17 = AWSChannelID("AWS17", "AWS1X", 6, 0)
    AWS18 = AWSChannelID("AWS18", "AWS1X", 7, 0)
    AWS21 = AWSChannelID("AWS21", "AWS2X", 8, 1)
    AWS31 = AWSChannelID("AWS31", "AWS3X", 9, 2)
    AWS32 = AWSChannelID("AWS32", "AWS3X", 10, 2)
    AWS33 = AWSChannelID("AWS33", "AWS3X", 11, 2)
    AWS34 = AWSChannelID("AWS34", "AWS3X", 12, 2)
    AWS35 = AWSChannelID("AWS35", "AWS3X", 13, 2)
    AWS36 = AWSChannelID("AWS36", "AWS3X", 14, 2)
    AWS41 = AWSChannelID("AWS41", "AWS4X", 15, 3)
    AWS42 = AWSChannelID("AWS42", "AWS4X", 16, 3)
    AWS43 = AWSChannelID("AWS43", "AWS4X", 17, 3)
    AWS44 = AWSChannelID("AWS44", "AWS4X", 18, 3)

    @staticmethod
    def from_name(name):
        for channel in AWSChannel:
            if channel.value.name == name:
                return channel
        raise ValueError(f"Channel {name} not found.")

    @staticmethod
    def from_index0(index0):
        for channel in AWSChannel:
            if channel.value.index0 == index0:
                return channel
        raise ValueError(f"Channel with 0 based index {index0} not found.")


@dataclass
class ObservationCase:
    long_name: str
    timerange: slice


# TODO: Move to a json file...
OBSERVATION_CASES = {
    "2025-01-01-swirls": ObservationCase(
        timerange=slice("2025-01-01T05:00:00", "2025-01-01T05:04:00"),
        long_name="Larger cloud system with internal swirls",
    ),
    "2025-01-28-test1": ObservationCase(
        timerange=slice("2025-01-28T18:38:00", "2025-01-28T18:41:00"),
        long_name="cloud?",
    ),
    "2025-01-12-dikeledi": ObservationCase(
        timerange=slice("2025-01-12T07:46:40", "2025-01-12T07:51:10"),
        long_name="dikeledi",
    ),
    "2025-02-02-midlat-uniform": ObservationCase(
        timerange=slice("2025-02-02T03:16:00", "2025-02-02T03:20:00"),
        long_name="midlatitude, uniform humidity, no clouds",
    ),
    "2025-02-02-midlat2-land": ObservationCase(
        timerange=slice("2025-02-02T19:21:00", "2025-02-02T19:24:00"),
        long_name="midlatitude, humid with dry streak mixed in. Has ground and ocean",
    ),
    "2025-02-02-midlat3-ocean": ObservationCase(
        timerange=slice("2025-02-02T20:58:00", "2025-02-02T21:00:00"),
        long_name="midlatitude, uniform patch over ocean",
    ),
    "2025-02-02-midlat4-thin-clouds-over-ocean": ObservationCase(
        timerange=slice("2025-02-02T22:36:00", "2025-02-02T22:39:00"),
        long_name="Midlatitude, thin clouds over ocean",
    ),
    "2025-02-03-midlat1-italy": ObservationCase(
        timerange=slice("2025-02-03T09:50:00", "2025-02-03T09:59:00"),
        long_name="Midlatitude, swirl of humidity, some clouds",
    ),
}


def augment_with_basic_qa_flag(ds: xr.Dataset) -> xr.Dataset:
    """Add QA flag variable for basic checks of data"""

    # brightness temperature values within range
    flag_bad_ta_value = (
        (ds.aws_toa_brightness_temperature < 50)
        | (ds.aws_toa_brightness_temperature > 350)
    ).any(dim=["n_channels", "n_fovs"])
    # TODO: Consider the new flags in the L1B file, and add them here as well.
    # NOTE: using a nan check instead of isnull. It appears to cause a segfault in this usecase.
    flag_bad_latlon = ((ds.aws_lat == np.nan) | (ds.aws_lon == np.nan)).any(
        dim=["n_geo_groups", "n_fovs"]
    )
    flag_bad_satellite_altitude = (ds.satellite_altitude < 400) | (
        ds.satellite_altitude > 800
    )
    flag_bad_counts = ds.L1B_quality_flag != 1.0

    flags_qa = (
        flag_bad_ta_value
        | flag_bad_satellite_altitude
        | flag_bad_latlon
        | flag_bad_counts
    )

    ds["flag_bad_data"] = xr.DataArray(
        data=flags_qa,
        dims=["n_scans"],
        attrs={
            "units": "1",
            "flag_values": "0, 1",
            "flag_meanings": "Scan has no deteceted bad values, Scan has bad value(s)",
            "long_name": "Quality flag for scan. 0 if values in scan pass basic QA checks",
        },
    )

    return ds


def get_file_l1b(filename) -> Path:
    """Get full path to a specific L1B file in default data directory."""

    filepath = Path(DATADIR_L1B) / filename

    if not filepath.exists():
        raise ValueError(
            f"Could not find file `{filepath}`. Is AWSPROCESSING_DATADIR properly set?"
        )

    return filepath


def filter_l1b_filepaths_keep_newest_processing_time(filepaths_sorted, pattern):
    """Return filepaths with unique start and end times for the latest processing time

    :param filepaths_sorted: Alphanumerically asc sorted list of filepaths to filter
    :param pattern: The pattern used to parse the datetime range from the filename

    :return: List of filepaths that have unique start and end times for the latest processing time.
    """
    filtered_paths = []

    num_filepaths = len(filepaths_sorted)
    previous_timerange = None
    for i, filepath in enumerate(filepaths_sorted):
        parsed = parse.parse(pattern, Path(filepath).name)
        if parsed is None:
            continue  # skip this entry

        current_timerange = (parsed["start_time"], parsed["end_time"])
        if (
            i > 0 and previous_timerange != current_timerange
        ) or i == num_filepaths - 1:
            filtered_paths.append(filepath)

        previous_timerange = current_timerange

    return filtered_paths


def filter_l1b_filepaths_based_on_timerange(
    filepaths, pattern, timerange, fully_contain=False
):
    """Return filepaths that contain any part of the given timerange


    :param filepath: List of filepaths to filter
    :param pattern: The pattern used to parse the datetime range from the filename
    :param timerange: The timerange to filter the filepaths with.
    :param fully_contain: Only return filepaths that fully contain the timerange, instead of files that have any part of the timerange.

    :return: List of filepaths that include the timerange or fully include the timerange.
    """
    timerange = timeslice_cast(timerange)
    filtered_paths = []
    for filepath in filepaths:
        parsed = parse.parse(pattern, Path(filepath).name)
        if parsed is None:
            continue  # skip this entry

        start_time = pd.Timestamp(parsed["start_time"])  # Not necessary?
        end_time = pd.Timestamp(parsed["end_time"])
        if fully_contain:
            if start_time <= timerange.start and end_time >= timerange.stop:
                filtered_paths.append(filepath)
        elif start_time <= timerange.stop and end_time >= timerange.start:
            filtered_paths.append(filepath)

    return filtered_paths


def get_files_l1b(
    glob_pattern: str = "W_NO-KSAT-Tromso*.nc", timerange=None
) -> Iterable[Path]:
    """Get a list of filepaths in the default data directory."""

    filepaths = list(Path(DATADIR_L1B).glob(glob_pattern))

    if not filepaths:
        raise ValueError(
            f"No input files found in `AWSPROCESSING_DATADIR={DATADIR_L1B}`."
        )

    filepaths = sort_l1b_filepaths_by_timerange(filepaths)

    filepaths = filter_l1b_filepaths_keep_newest_processing_time(
        filepaths, FILEPATTERN_L1B
    )

    if not timerange:
        return filepaths

    if timerange:
        filepaths = filter_l1b_filepaths_based_on_timerange(
            filepaths, FILEPATTERN_L1B, timerange
        )

    return filepaths


class FieldDimType(Enum):
    CHANNEL_SCAN_FOV = 1
    SCAN = 2
    GROUP_SCAN_FOV = 3


@xr.register_dataset_accessor("geo")
class ChannelGeoAccessor:
    def __init__(self, xarray_ds):
        self._ds = xarray_ds

    def toa_tb(self, channel_names_in_group):
        """Return DataArray of TOA Tb for channel with corresponding lat-lon coordinates"""
        if isinstance(channel_names_in_group, list):
            # verify that all channels are in the same group
            chs = [
                AWSChannel.from_name(channel_name).value
                for channel_name in channel_names_in_group
            ]
            assert all(
                ch.geo_group_name == chs[0].geo_group_name for ch in chs
            ), "Requested channels are not in the same group"
            ch = chs[0]
        else:
            ch = AWSChannel.from_name(channel_names_in_group).value

        da = self._ds.aws_toa_brightness_temperature.sel(
            n_channels=channel_names_in_group
        )
        da.coords["lat"] = self._ds.aws_lat.sel(n_geo_groups=ch.geo_group_name)
        da.coords["lon"] = self._ds.aws_lon.sel(n_geo_groups=ch.geo_group_name)
        return da


def sort_l1b_filepaths_by_timerange(filepaths):
    parsed_filepaths = [
        parse.parse(FILEPATTERN_L1B, filepath.name) for filepath in filepaths
    ]
    parsed_filepaths = [p for p in parsed_filepaths if p is not None]

    sorted_filepaths = sorted(
        filepaths,
        key=lambda x: parsed_filepaths[filepaths.index(x)]["end_time"],
        reverse=False,
    )
    return sorted_filepaths


def load_multiple_files_l1b(filepaths: Iterable[str], apply_fixes=True):
    """
    Load multiple relevant fields for multiple L1B files for specified channels into a single dataset

    This function serves as a wrapper to repeatedly call `load_single_l1b` for the given filepaths
    and then concatenate them along the n_scans dimension.

    :param filepaths: List or other iterable of filepaths to AWS L1B .nc files provided by EUMETSAT
    :param channels: List of channels to load
    :param apply_fixes: Correct/filter known issues, see function _apply_datafixes for details.

    :return: Dataset with a subset of the relevant fields in the L1B data.
    """
    last_timestamp = np.datetime64(0, "Y")
    datasets = []
    for filepath in tqdm(filepaths):
        ds_single = load_single_l1b(filepath, apply_fixes=False)

        # Trim any overlap
        # TODO: Remove this, and treat it like the datasets instead with L1BSequence as input?
        if ds_single.time[0] < last_timestamp:
            # Find index of first scan that is after the last scan of the previous file
            i_first_scan = np.where(ds_single.time > last_timestamp)[0][0]
            ds_single = ds_single.isel(n_scans=slice(i_first_scan, None))

        last_timestamp = ds_single.time[-1].values
        datasets.append(ds_single)

    print(f"loaded {len(datasets)} files")

    # Combine the datasets along n_scans
    ds = xr.concat(datasets, "n_scans")

    if apply_fixes:
        ds = _apply_datafixes(ds)  # Apply fixes in one go

    return ds


def load_single_l1b(filepath: str, apply_fixes=True):
    """Load relevant fields from L1B file for specified channels into a single dataset

    :param filepath: path to a single granule .zip AWS L1B file provided by EUMETSAT.
    :param channels: List of channels to load
    :param apply_fixes: Correct/filter known issues, see function _apply_datafixes for details.

    :return: Dataset with a subset of the relevant fields in the L1B data.
    """

    dt = xr.open_datatree(filepath, chunks={"n_scans": "auto"}, decode_timedelta=True)

    # Construct a new dataset with the relevant fields
    fields_to_extract = [
        (
            "data/calibration",
            "aws_toa_brightness_temperature",
            FieldDimType.CHANNEL_SCAN_FOV,
        ),
        ("data/navigation", "time_startscan_utc_earthview", FieldDimType.SCAN),
        ("data/navigation", "aws_lat", FieldDimType.GROUP_SCAN_FOV),
        ("data/navigation", "aws_lon", FieldDimType.GROUP_SCAN_FOV),
        ("data/navigation", "aws_terrain_elevation", FieldDimType.GROUP_SCAN_FOV),
        ("data/navigation", "aws_satellite_zenith_angle", FieldDimType.GROUP_SCAN_FOV),
        ("data/navigation", "aws_satellite_azimuth_angle", FieldDimType.GROUP_SCAN_FOV),
        ("data/navigation", "aws_surface_type", FieldDimType.GROUP_SCAN_FOV),
        ("data/navigation", "satellite_altitude", FieldDimType.SCAN),
        ("quality", "L1B_quality_flag", FieldDimType.SCAN),
        (
            "data/processing_information",
            "aws_l1b_processing_info_chan",
            FieldDimType.SCAN,
        ),
        ("data/processing_information", "aws_scanline_quality", FieldDimType.SCAN),
        ("data/processing_information", "aws_navigation_status", FieldDimType.SCAN),
        ("data/processing_information", "aws_surface_flag", FieldDimType.SCAN),
    ]

    extracted_data = {}
    for group_name, field_name, _ in fields_to_extract:
        extracted_data[field_name] = dt[group_name][field_name]

    # Create new dataset from extracted data
    new_ds = xr.Dataset(
        extracted_data,
        coords={
            "n_channels": ("n_channels", [ch.value.name for ch in AWSChannel]),
            "n_geo_groups": ("n_geo_groups", ["AWS1X", "AWS2X", "AWS3X", "AWS4X"]),
        },
    )

    # Use time as coordinate for scan, wich requires monotonically increasing time.
    # However, some timestamps are bad. We therefore patch them here for the coordinate,
    # the real timestamp can still be accessed via time_startscan_utc_earthview.
    coord_array = new_ds.time_startscan_utc_earthview.values
    ibad_times = np.where(coord_array < pd.Timestamp("2024-01-01T00:00:00"))[0]

    scan_ns = 1_190_000_000  # 1.19 second per scan approx
    # Patch first scan time
    if len(ibad_times) > 0 and ibad_times[0] == 0:
        ibad_time = ibad_times[0]
        ifwd = 1
        while (ibad_times + ifwd) in ibad_times:
            ifwd += 1
        coord_array[ibad_time] = coord_array[ibad_time + ifwd] - ifwd * scan_ns
        ibad_times = ibad_times[1:]

    # Patch remaining scan times
    for ibad_time in ibad_times:
        assert ibad_time > 0
        irev = 1
        while (ibad_time - irev) in ibad_times:
            irev += 1
        coord_array[ibad_time] = coord_array[ibad_time - irev] + irev * scan_ns

    new_ds.coords["time"] = ("n_scans", coord_array)
    new_ds = new_ds.set_xindex("time")

    # Attributes
    new_ds.attrs["source_file"] = Path(filepath).name

    if apply_fixes:
        parsed = parse.parse(FILEPATTERN_L1B, Path(filepath).name)
        if parsed is None:
            raise ValueError(
                f"Could not parse filename `{filepath}`. Is the filename in the expected format?"
            )
        if pd.Timestamp(parsed["processing_time"]) < pd.Timestamp("2025-03-13"):
            new_ds = _apply_datafixes_pre_20250313(new_ds)
        else:
            new_ds = _apply_datafixes(new_ds)

    # Add basic QA flags
    new_ds = augment_with_basic_qa_flag(new_ds)

    # Ensure that dimensions match the original L1b file!
    for dim in ["n_scans", "n_geo_groups", "n_channels", "n_fovs"]:
        assert (
            new_ds.sizes[dim] == dt["data/calibration"].sizes[dim]
        ), "Expect loaded dataset to match dimensions of original l1b file."

    # Add index coordinate for the original n_scans, n_fovs dimensions
    # So that if dataset is cropped, we can refer back to the original L1b file.
    new_ds["l1b_index_scans"] = xr.Variable(
        "n_scans",
        np.arange(new_ds.sizes["n_scans"], dtype=np.uint32),
        attrs={
            "units": "1",
            "long_name": "Index along the n_scans dimension in orignal L1B file (along-track). In the L2 file, scan time is sued as the along-track dimension in order to facilitate easier concatination and selection based on timeranges.",
        },
    )
    new_ds["l1b_index_fovs"] = xr.Variable(
        "n_fovs",
        np.arange(new_ds.sizes["n_fovs"], dtype=np.uint32),
        attrs={
            "units": "1",
            "long_name": "Index along n_fovs dimension in orignal L1B file",
        },
    )

    dt.close()
    return new_ds


def _apply_datafixes_pre_20250313(ds):
    """Adjust data according to known limitations, for data before 2025-03-13"""

    ds_fixed = ds

    # Bias corrections
    # For group AWS1X and AWS2X, offsets from O-B analysis of a "12hr window of AWS data on 20th January 2025"
    # (From correspondance with Dave Duncan 2025-01-28: dave.txt)
    N_FOVS = 145
    assert ds.sizes["n_fovs"] == N_FOVS
    aws3x_slope = np.arange(N_FOVS) * 0.9 / 120 - 0.54
    bias_corrections = {
        "AWS11": 2.092867,
        "AWS12": 2.303491,
        "AWS13": 1.953094,
        "AWS14": 2.011740,
        "AWS15": 1.655475,
        "AWS16": 1.711089,
        "AWS17": 1.599472,
        "AWS18": 1.192258,
        "AWS21": 1.755642,
        "AWS31": aws3x_slope + 1.923884,
        "AWS32": aws3x_slope + 1.662608,
        "AWS33": aws3x_slope + 1.409902,
        "AWS34": aws3x_slope + 1.687364,
        "AWS35": aws3x_slope + 0.782909,
        "AWS36": aws3x_slope + 1.407531,
        "AWS41": 0.699264,
        "AWS42": 0.613462,
        "AWS43": 0.681531,
        "AWS44": 0.899234,
    }

    offset_matrix = np.zeros([ds.sizes["n_fovs"], ds.sizes["n_channels"]])
    for channel, bias in bias_corrections.items():
        offset_matrix[:, AWSChannel.from_name(channel).value.index0] = bias

    ds_fixed["aws_toa_brightness_temperature"].data += offset_matrix

    # NOTE: Channel AWS35, has an additional bias dependence on laitude, as seen in the "Statistics for Radiances from AWS Channel=14" figure in Dave's email.
    # We find the variation by fitting the sinusoidal function to the latitudes and add to the brightness temperature.
    # lats = np.array([60, 30, 0, -30, -60])
    # ta = np.array([0.58, 0.87, 1.16, 0.87, 0.29])
    # A * np.cos(2*np.deg2rad(x))) + B
    # Fitted parameters: A=-0.4672222218344105, B=-0.6605555555604291
    # The zero mean over -90 to 90 degrees is latitude is taken by skipping B. We use the average bias-correction from Dave's email for the offset
    latude_cos_variation = -0.4672 * da.cos(
        2.0 * da.deg2rad(ds_fixed.aws_lat.sel(n_geo_groups="AWS3X").data)
    )
    ds_fixed["aws_toa_brightness_temperature"].loc[
        dict(n_channels="AWS35")
    ].data += latude_cos_variation

    return ds_fixed


def _apply_datafixes(ds):
    """Adjust data according to known limitations

    NOTE: Ensure input ds is data that has been just been loaded from disk.
    As to not risk applying these fixes multiple times.

    The data is adjusted in the following ways:
    - Adjust channel temperature values according to known biases.

    :param ds: Dataset corresponding to a single .nc file, that has not been processed.

    :return: Dataset with various fixes applied
    """

    ds_fixed = ds

    # Bias corrections (O-B)
    # Offsets from correspondance with David Duncan 2025-04-24. O-B After L1B processor update for side-lobe corrections
    N_FOVS = 145
    assert ds.sizes["n_fovs"] == N_FOVS
    aws3x_slope = np.arange(N_FOVS) * 0.9 / 120 - 0.54
    bias_corrections = {
        "AWS11": 1.164286,
        "AWS12": 1.293120,
        "AWS13": 0.883213,
        "AWS14": 1.020994,
        "AWS15": 0.860618,
        "AWS16": 1.000565,
        "AWS17": 0.989464,
        "AWS18": 0.748243,
        "AWS21": -3.428536,
        "AWS31": aws3x_slope + 0.570022,
        "AWS32": aws3x_slope - 0.121587,
        "AWS33": aws3x_slope - 0.261025,
        "AWS34": aws3x_slope - 0.179699,
        "AWS35": aws3x_slope - 0.922225,
        "AWS36": aws3x_slope - 0.490903,
        "AWS41": 0.499461,
        "AWS42": 0.418686,
        "AWS43": 0.591923,
        "AWS44": 0.303710,
    }

    offset_matrix = np.zeros([ds.sizes["n_fovs"], ds.sizes["n_channels"]])
    for channel, bias in bias_corrections.items():
        offset_matrix[:, AWSChannel.from_name(channel).value.index0] = bias

    ds_fixed["aws_toa_brightness_temperature"].data += offset_matrix

    # NOTE: Channel AWS35, has an additional bias dependence on laitude, as seen in the "Statistics for Radiances from AWS Channel=14" figure in Dave's email.
    # We find the variation by fitting the sinusoidal function to the latitudes and add to the brightness temperature.
    # lats = np.array([60, 30, 0, -30, -60])
    # ta = np.array([0.58, 0.87, 1.16, 0.87, 0.29])
    # A * np.cos(2*np.deg2rad(x))) + B
    # Fitted parameters: A=-0.4672222218344105, B=-0.6605555555604291
    # The zero mean over -90 to 90 degrees is latitude is taken by skipping B. We use the average bias-correction from Dave's email
    latude_cos_variation = -0.4672 * da.cos(
        2.0 * da.deg2rad(ds_fixed.aws_lat.sel(n_geo_groups="AWS3X").data)
    )
    ds_fixed["aws_toa_brightness_temperature"].loc[
        dict(n_channels="AWS35")
    ].data += latude_cos_variation

    return ds_fixed
