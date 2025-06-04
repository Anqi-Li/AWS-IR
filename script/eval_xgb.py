# %%
import xgboost as xgb
import xarray as xr
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt
from aws_ir.ml_loading import get_ml_ds, get_ml_filepaths_by_timerange
from sklearn.metrics import mean_squared_error
import random
import pandas as pd
from scipy.stats import binned_statistic
import json

# %%
model_tag = "2025-04-20days"

timerange = slice("2025-04-01T00:00:00", "2025-04-20T00:00:00")
filepaths = get_ml_filepaths_by_timerange(timerange=timerange)

# 80% 20% training and testing data
random.seed(4)
random.shuffle(filepaths)
split_index = int(len(filepaths) * 0.8)
filepaths_train = filepaths[:split_index]
filepaths_test = filepaths[split_index:]

# %% Evaluation data
print("Loading test data...")
start = datetime.now()
Xy_test = get_ml_ds(filepaths=filepaths_test)
X_test = Xy_test.aws_toa_brightness_temperature.T
y_test = Xy_test.Tb
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=X_test.n_channels.values.tolist())
print("Load data", datetime.now() - start)

# %% Load trained model
model = xgb.Booster()
model.load_model("../data/model/xgb_regressor_{}.json".format(model_tag))
print("Model loaded from file")

# %% load saved evaluation result from file
with open("../data/model/evals_result_{}.json".format(model_tag), "r") as f:
    evals_result_ = json.load(f)
print("Evaluation results loaded from file")

# %% Plot training and validation RMSE
plt.figure(figsize=(6, 5))
df = pd.DataFrame(evals_result_)
plt.plot(df["train"]["rmse"])
plt.plot(df["validation"]["rmse"])
plt.xlabel("Boosting rounds")
plt.ylabel("RMSE")
plt.title("XGBoost Training and Validation RMSE")
plt.legend(["Train", "Validation"])
plt.tight_layout()

# %% Plot feature importance
plt.figure(figsize=(10, 6))
xgb.plot_importance(
    model,
    importance_type="weight",  # weight or "gain" or "cover"
    max_num_features=20,  # Limit to top 20 features
    title="Feature Importance by weight",
    # xlabel="F score",
    ylabel="Features",
)
plt.tight_layout()
plt.show()


# %% Plot predicted vs true values histogram
y_pred = model.predict(dtest)


# %% Plot conditional probability
def per90(a):
    return np.percentile(a, 90)


def per10(a):
    return np.percentile(a, 10)


n_bins = 30
bin_edges = np.linspace(180, 330, n_bins + 1)
bin_mid = (bin_edges[:-1] + bin_edges[1:]) / 2
bin_means, bin_edges, binnumber = binned_statistic(y_test, y_pred, statistic="mean", bins=bin_edges)
bin_per90, _, _ = binned_statistic(y_test, y_pred, statistic=per90, bins=bin_edges)
bin_per10, _, _ = binned_statistic(y_test, y_pred, statistic=per10, bins=bin_edges)
plt.plot(bin_mid, bin_means, label="Mean", c="k")
plt.plot(bin_mid, bin_per90, label="90th percentile")
plt.plot(bin_mid, bin_per10, label="10th percentile")

h_joint_test_pred, _, _ = np.histogram2d(y_test, y_pred, bins=bin_edges, density=True)
h_test, _ = np.histogram(y_test, bins=bin_edges, density=True)
h_conditional = h_joint_test_pred / h_test.reshape(-1, 1)
h_conditional_nan = np.where(h_conditional > 0, h_conditional, np.nan)

c = plt.contourf(bin_mid, bin_mid, h_conditional_nan.T, cmap="Blues", vmin=0, vmax=0.14)
# c = plt.pcolormesh(bin_edges, bin_edges, h_conditional_nan.T, cmap="Blues", vmin=0, vmax=0.14)
plt.colorbar(c, label="Probability density")

# Add diagonal line
plt.plot(
    [bin_mid[0], bin_mid[-1]],
    [bin_mid[0], bin_mid[-1]],
    "r--",
    label="True = Predicted",
)
plt.legend()
plt.xlim(bin_edges[0], bin_edges[-1])
plt.ylim(bin_edges[0], bin_edges[-1])
plt.xlabel("True [K]")
plt.ylabel("Predicted [K]")
plt.title("P(Predicted | True)")
plt.show()

# %%
np.nansum(h_conditional.T * np.diff(bin_edges), axis=0)

# %% Plot joint probability
c = plt.contour(
    bin_mid,
    bin_mid,
    h_joint_test_pred.T,
    levels=np.logspace(-5, -2, 10),
)
cbar = plt.gcf().colorbar(c)
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
plt.title("Predicted vs True values histogram")
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
y.lon.attrs["long_name"] = "Longitude"
y.lat.attrs["long_name"] = "Latitude"

# %% plot a subset of the scene
y_subset = y.isel(time=slice(2000, 2200)).load()
y_subset = y_subset.assign_coords(
    lat=y_subset.lat.interpolate_na("time", fill_value="extrapolate"),
    lon=y_subset.lon.interpolate_na("time", fill_value="extrapolate"),
)

fig, axes = plt.subplots(1, 3, sharex=True, sharey=True)
kwargs = dict(x="lon", y="lat", vmin=200, vmax=300)
y_subset.y_true.plot(ax=axes[0], **kwargs)
y_subset.y_pred.plot(ax=axes[1], **kwargs)
(y_subset.y_pred - y_subset.y_true).plot(ax=axes[2], x="lon", y="lat", vmin=-20, vmax=20, cmap="RdBu_r")
axes[0].set_title("True")
axes[1].set_title("Predicted")
axes[2].set_title("Residuals (Predicted - True)")
axes[1].set_ylabel("")
axes[2].set_ylabel("")
fig.suptitle("Subset of scene {} ".format(y_subset.time.min().values))
fig.tight_layout()

# %%
from aws_ir import cpcir_loading, aws_loading
from datetime import timedelta

timerange = slice(y_subset.time.min(), y_subset.time.max())
files_cpcir = cpcir_loading.get_cpcir_fileset(
    timerange=timerange,
)
ds_cpcir = cpcir_loading.get_cpcir_ds(files_cpcir)

files_aws = aws_loading.get_files_l1b(timerange=timerange)
ds_aws = aws_loading.load_multiple_files_l1b(files_aws, apply_fixes=True)
ds_aws = ds_aws.sel(time=timerange, n_fovs=slice(40, 100))

# %%
fig, axes = plt.subplots(1, 3, sharex=True, sharey=True)
kwargs = dict(x="lon", y="lat", vmin=200, vmax=300)

ds_cpcir.sel(
    time=y_subset.time[0],
    method="nearest",
    tolerance=timedelta(minutes=30),
).sel(
    lat=slice(y_subset.lat.min(), y_subset.lat.max()),
    lon=slice(y_subset.lon.min(), y_subset.lon.max()),
).Tb.plot(ax=axes[0], **kwargs)
y_subset.y_true.plot(ax=axes[1], **kwargs)
y_subset.y_pred.plot(ax=axes[2], **kwargs)
axes[0].set_title("Org cpcir")
axes[1].set_title("True \n(interpolated cpcir on aws swath)")
axes[2].set_title("Predicted")
fig.tight_layout()

# %%

# %%
