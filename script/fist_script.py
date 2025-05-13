# %% import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

import os
import sys
import xgboost as xgb
from datetime import datetime

# getting the name of the directory where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
# Getting the parent directory name where the current directory is present.
parent = os.path.dirname(current)
# adding the parent directory to the sys.path.
sys.path.append(parent)

from src import aws_loading, cpcir_loading, aws_remapping
from src.combine import align_cpcir, package_ml_xy

model_tag = 'ten_days'
print(model_tag)

# %% Load AWS data
timerange = slice("2025-04-01T00:00:00", "2025-04-10T00:00:00")
files_aws = aws_loading.get_files_l1b(
    timerange=timerange,
)
ds_aws = aws_loading.load_multiple_files_l1b(files_aws, apply_fixes=True)
ds_aws = ds_aws.sel(time=timerange, n_fovs=slice(40, 100))
ds_aws = ds_aws.where((ds_aws.flag_bad_data == 0).compute(), drop=True)

# %% remap AWS data
ds_aws_remapped = aws_remapping.remap_interp(
    ds_aws,
    remap_to_ch="AWS33",
    method="nearest",
    fill_distance=True,
)
# remove points outside the CPCIR latitude coverage
ds_aws_remapped = ds_aws_remapped.where(
    (ds_aws_remapped.aws_lat.pipe(abs) < 59.981808).compute(),
    drop=True,
)
# %% Load CPCIR data
files_cpcir = cpcir_loading.get_cpcir_fileset(
    timerange=slice(ds_aws.time.min(), ds_aws.time.max()),
)
ds_cpcir = cpcir_loading.get_cpcir_ds(files_cpcir)

# %% colocate CPCIR to AWS data
ds_cpcir_aligned = align_cpcir(ds_cpcir, ds_aws_remapped)

# %%
print("Loading data...")
X_train, y_train = package_ml_xy(ds_aws_remapped, ds_cpcir_aligned)

start = datetime.now()
dtrain = xgb.DMatrix(X_train, label=y_train)
print("Load training data", datetime.now() - start)

# %% Fit model
print("Training XGBoost regression model...")
start = datetime.now()

# Define parameters for the XGBoost model
params = {
    "objective": "reg:squarederror",  # Regression objective
    "eval_metric": "rmse",  # Root Mean Square Error as evaluation metric
    "eta": 0.1,  # Learning rate
    "max_depth": 6,  # Maximum depth of a tree
    "subsample": 0.8,  # Subsample ratio of the training instances
    "colsample_bytree": 0.8,  # Subsample ratio of columns when constructing each tree
    "seed": 42,  # Random seed for reproducibility
}

# Train the model
num_boost_round = 100  # Number of boosting rounds
model = xgb.train(params, dtrain, num_boost_round)

print("Model training completed in", datetime.now() - start)

# %% Save the model to a file
model.save_model("./data/xgb_regressor_{}.json".format(model_tag))
print("Model saved")

#%%

# y_pred = model.predict(data=dtrain)

# h = np.histogram2d(y_train.squeeze(), y_pred.squeeze(), bins=(30,30), density=True,)
# plt.contour(
#     h[1][:-1],
#     h[2][:-1],
#     h[0].T,
#     # levels=np.logspace(-4, -2, 8),
# )