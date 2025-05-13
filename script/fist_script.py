# %% import libraries
from importlib.metadata import files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

import os
import sys

# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)

# adding the parent directory to
# the sys.path.
sys.path.append(parent)

from src import aws_loading, cpcir_loading

# %%
timerange = slice("2025-04-01T00:00:00", "2025-04-01T01:00:00")
files_aws = aws_loading.get_files_l1b(
    timerange=timerange,
)
ds_aws = aws_loading.load_multiple_files_l1b(files_aws, apply_fixes=True)

# %%
files_cpcir = cpcir_loading.get_cpcir_fileset(
    timerange=slice(ds_aws.time.min(), ds_aws.time.max()),
)
ds_cpcir = cpcir_loading.get_cpcir_ds(files_cpcir)

# %%
