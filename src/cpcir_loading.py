import xarray as xr
import pandas as pd
import parse
from pathlib import Path
import sshfs
from .aws_loading import timeslice_cast

FILEPATTERN_CPCIR = "merg_{time:%Y%m%d%H}_4km-pixel.nc4"
PATH_CPCIR = "/scratch/li/cpcir"


def get_cpcir_fileset(timerange=None):
    """
    Get the fileset for the given timerange.
    Args:
        timerange (slice): The time range to filter the files.
    Returns:
        list: A list of file paths that match the timerange.
    """

    # get all files in the directory PATH_CPCIR
    fileset = sorted(Path(PATH_CPCIR).glob("*.nc4"))

    # Filter the fileset for the given timerange.
    timerange = timeslice_cast(timerange)
    filtered_paths = []
    for filepath in fileset:
        parsed = parse.parse(FILEPATTERN_CPCIR, Path(filepath).name)
        if parsed is None:
            continue  # skip this entry

        file_datetime = pd.Timestamp(parsed["time"])
        start_time = file_datetime - pd.Timedelta("30min")
        end_time = file_datetime + pd.Timedelta("30min")
        if start_time <= timerange.stop and end_time >= timerange.start:
            filtered_paths.append(filepath)

    # fileset = [fs.get_mapper(file_path) for file_path in filtered_paths]
    return filtered_paths


def get_cpcir_ds(fileset):
    ds = xr.open_mfdataset(fileset)
    return ds
