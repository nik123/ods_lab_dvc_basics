import json

import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error


with open("data/weights.joblib", 'rb') as f:
    model = pickle.load(f)

test_set = pd.read_csv("data/train.csv")
test_X = test_set.drop(["median_house_value"], axis=1, inplace=False)
test_y = test_set["median_house_value"]

predictions = model.predict(test_X)
rmse = np.sqrt(mean_squared_error(test_y, predictions))

with open("data/metrics.json", "w") as f:
    json.dump({"rmse": rmse}, f)
    f.write("\n")  # Add newline because json.dump does not
