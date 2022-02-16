import pandas as pd

import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

with open("params.yaml") as f:
    params = yaml.safe_load(f)

csv_path = "data/housing.csv"
housing = pd.read_csv(csv_path)

median = housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median, inplace=True)

housing_cat = housing[["ocean_proximity"]]
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing.drop(["ocean_proximity"], axis=1, inplace=True)
housing["ocean_proximity"] = housing_cat_encoded


numerical_columns = [
    "longitude",
    "latitude",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
    "housing_median_age"
]
scaler = StandardScaler()
housing_scaled = scaler.fit_transform(housing[numerical_columns])
housing.drop(numerical_columns, axis=1, inplace=True)
for i, c in enumerate(numerical_columns):
    housing[c] = housing_scaled[:, i]

train_set, test_set = train_test_split(
    housing,
    test_size=params["prepare"]["split"],
    random_state=params["prepare"]["seed"],
)

train_set.to_csv("data/train.csv")
test_set.to_csv("data/test.csv")
