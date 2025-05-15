# %%
import xgboost as xgb
import xarray as xr
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt
from aws_ir.ml_loading import get_ml_ds, get_ml_filepaths_by_timerange
from sklearn.metrics import mean_squared_error
import random

# %%
model_tag = "test"

timerange = slice("2025-04-01T00:00:00", "2025-04-10T00:00:00")
filepaths = get_ml_filepaths_by_timerange(timerange=timerange)

# 80% 20% training and testing data
random.shuffle(filepaths)
split_index = int(len(filepaths) * 0.8)
filepaths_train = filepaths[:split_index]
filepaths_test = filepaths[split_index:]

# %% Training data
print("Loading traning data...")
start = datetime.now()
Xy_train = get_ml_ds(filepaths=filepaths_train)
X_train = Xy_train.aws_toa_brightness_temperature.T
y_train = Xy_train.Tb
dtrain = xgb.DMatrix(X_train, label=y_train)
print("Load data", datetime.now() - start)

# %% Evaluation data
print("Loading test data...")
start = datetime.now()
Xy_test = get_ml_ds(filepaths=filepaths_test)
X_test = Xy_test.aws_toa_brightness_temperature.T
y_test = Xy_test.Tb
dtest = xgb.DMatrix(X_test, label=y_test)
print("Load data", datetime.now() - start)

# %% Train model
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
    # "tree_method": "gpu_hist",
}

# Train the model
evals_result = {}
model = xgb.train(
    params,
    dtrain,
    num_boost_round=100,  # Number of boosting rounds
    evals=[(dtrain, "train"), (dtest, "validation")],
    early_stopping_rounds=10,
    verbose_eval=10,
    # evals_result=evals_result,
    # xgb_model=,
)

print("Model training completed in", datetime.now() - start)

# %% Save the model to a file
model.save_model("../data/model/xgb_regressor_{}.json".format(model_tag))
print("Model saved")

# %% Plot predicted y
y_pred = model.predict(dtest)

h = np.histogram2d(y_test, y_pred, bins=(30, 30), density=True)
CS = plt.contour(
    h[1][:-1],
    h[2][:-1],
    h[0].T,
    # levels=np.logspace(-4, -2, 8),
)
cbar = plt.gcf().colorbar(CS)
# Add diagonal line
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--", label="True = Predicted")

plt.text(
    0.1,
    0.9,
    f"RMSE {np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_test)):.2f} K",
    transform=plt.gca().transAxes,
)
plt.xlabel("True")
plt.ylabel("Predicted")
plt.title("{} - {}".format(y_test.time.min().values, y_test.time.max().values))
plt.show()
# %% unstack the fov dimension and reconstruct the scene
y = (
    y_test.to_dataset(name="y_true")
    .assign(
        y_pred=xr.DataArray(
            y_pred,
            coords=y_test.coords,
            dims=y_test.dims,
            name="y_pred",
            attrs=y_test.attrs,
        )
    )
    .drop_duplicates("n_samples")
    .unstack("n_samples")
)

# %%
y_subset = y.isel(time=slice(1500, 1800)).load()
y_subset = y_subset.assign_coords(
    lat=y_subset.lat.interpolate_na("time", fill_value="extrapolate"),
    lon=y_subset.lon.interpolate_na("time", fill_value="extrapolate"),
)

fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
kwargs = dict(x="lon", y="lat", vmin=240, vmax=300)
y_subset.y_true.plot(ax=axes[0], **kwargs)
y_subset.y_pred.plot(ax=axes[1], **kwargs)
axes[0].set_title("True")
axes[1].set_title("Predicted")
fig.tight_layout()

# %%
