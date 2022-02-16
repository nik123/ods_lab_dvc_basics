import pickle
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor

with open("params.yaml") as f:
    params = yaml.safe_load(f)

train_set = pd.read_csv("data/train.csv")
train_X = train_set.drop(["median_house_value"], axis=1, inplace=False)
train_y = train_set["median_house_value"]

model = RandomForestRegressor(
    n_estimators=params["train"]["n_estimators"],
    max_features=params["train"]["max_features"],
    random_state=params["train"]["seed"],
)
model.fit(train_X, train_y)

with open("data/weights.joblib", 'wb') as f:
    pickle.dump(model, f)
