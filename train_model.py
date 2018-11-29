"""
To train this model, in your terminal:
> python train_model.py
"""

from sklearn.externals import joblib
import pandas as pd
from sklearn.metrics import r2_score
from sklearn import linear_model

print('Loading the wine-review dataset')
df = pd.read_csv('C:\AI\\wine-reviews\winemag-data-130k-v2.csv', na_values="?")
data = df[["points", "price"]]

print('Training a LinearRegression')
data = data.dropna()
 # Treat the column price as our predictive objective
data_y = data["price"]
 # All other columns will be used as features when training our model
data_x = data.drop(["price"], axis=1)
regr = linear_model.LinearRegression()
regr.fit(data_x, data_y)

print('Exporting the trained model')
joblib.dump(regr, 'model/wine-review.joblib')
