#%%
import xarray as xr
import parse
from pathlib import Path
from aws_ir.utils import timeslice_cast
import pandas as pd

#%%
FILEPATTERN = 'aws_cpcir_{start:%Y%m%d%H%M%S}_{end:%Y%m%d%H%M%S}.nc'
PATH = '../data'

def get_ml_filepaths_by_timerange(timerange):
    timerange = timeslice_cast(timerange)

    filepaths_all = sorted(Path(PATH).glob("*.nc"))
    filepaths_valid = []
    for filepath in filepaths_all:
        parsed = parse.parse(FILEPATTERN, Path(filepath).name)
        if parsed is None:
            continue  # skip this entry
        start_time =  pd.Timestamp(parsed["start"])
        end_time = pd.Timestamp(parsed['end'])
        if start_time <= timerange.stop and end_time >= timerange.start:
            filepaths_valid.append(filepath)

    return filepaths_valid

    
    
def get_ml_ds(filepaths):
    Xy = xr.open_mfdataset(filepaths, combine='nested', concat_dim='n_samples')
    return Xy.set_xindex(['time', 'n_fovs'])
    