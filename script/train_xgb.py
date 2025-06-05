# %%
import xgboost as xgb
from datetime import datetime
from aws_ir.ml_loading import get_ml_ds, get_ml_filepaths_by_timerange
import random
import json

# %%
model_tag = "2025-03-14to2025-05-31_seed4"

timerange = slice("2025-04-01T00:00:00", "2025-04-05T23:59:00")
filepaths = get_ml_filepaths_by_timerange(timerange=timerange)

# 80% 20% training and testing data
random.seed(4)  # Set seed for reproducibility
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
dtrain = xgb.DMatrix(
    X_train,
    label=y_train,
    feature_names=X_train.n_channels.values.tolist(),
)
print("Load data", datetime.now() - start)

# %% Evaluation data
print("Loading test data...")
start = datetime.now()
Xy_test = get_ml_ds(filepaths=filepaths_test)
X_test = Xy_test.aws_toa_brightness_temperature.T
y_test = Xy_test.Tb
dtest = xgb.DMatrix(
    X_test,
    label=y_test,
    feature_names=X_test.n_channels.values.tolist(),
)
print("Load data", datetime.now() - start)

# %% Load previous model
model_tag_previous = model_tag  # or 'test
model_previous = xgb.Booster()
model_previous.load_model(f"../data/model/xgb_regressor_{model_tag_previous}.json")
print("Previous model loaded from file")

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
    num_boost_round=1000,  # Number of boosting rounds
    evals=[(dtrain, "train"), (dtest, "validation")],
    early_stopping_rounds=10,
    verbose_eval=10,
    evals_result=evals_result,
    xgb_model=model_previous,
)

print("Model training completed in", datetime.now() - start)

# %% load previous evaluation result from file
with open("../data/model/evals_result_{}.json".format(model_tag_previous), "r") as f:
    evals_result_previous = json.load(f)
print("Previous evaluation results loaded from file")
evals_result_previous["train"]["rmse"].extend(evals_result["train"]["rmse"])
evals_result_previous["validation"]["rmse"].extend(evals_result["validation"]["rmse"])
len(evals_result_previous["train"]["rmse"])

# %%
with open("../data/model/evals_result_{}.json".format(model_tag), "w") as f:
    json.dump(evals_result_previous, f)
print("Combined evaluation results saved")

# %% Save evaluation result to a file
# with open("../data/model/evals_result_{}.json".format(model_tag), "w") as f:
#     json.dump(evals_result, f)
# print("Evaluation results saved")

# %% Save the model to a file
model.save_model("../data/model/xgb_regressor_{}.json".format(model_tag))
print("Model saved")

# %%
