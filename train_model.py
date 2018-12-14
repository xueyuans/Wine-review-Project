"""
To train this model, in your terminal:
> python train_model.py
"""

from sklearn.externals import joblib
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

le_country = LabelEncoder()
le_province = LabelEncoder()
le_taster_name = LabelEncoder()

def encodeStr(df):
    # use the LabelEncoder function to convert the country to corresponding integers.
    le_country.fit(df['country'])
    df['country']= le_country.transform(df['country'])

    # use the LabelEncoder function to convert the province to corresponding integers.
    le_province.fit(df['province'])
    df['province']= le_province.transform(df['province'])

    # use the LabelEncoder function to convert the taster_name to corresponding integers.
    le_taster_name.fit(df['taster_name'])
    df['taster_name']= le_taster_name.transform(df['taster_name'])

    return df

def GoodRating(row):
    if row['points'] >= 92:
        val = 3
    elif row['points'] >= 86 and row['points'] < 92:
        val = 2
    else:
        val = 1
    return(val)

# Train models for prect1
print('Loading the wine-review dataset')
df = pd.read_csv('wine-reviews_data/winemag-data-130k-v2.csv', na_values="?")
df = df.where(df['price'] < 1500)

df = df.drop(['Unnamed: 0','description','taster_twitter_handle','title','region_1', 'region_2','designation','variety','winery'],1)
df = df.dropna()
df = encodeStr(df)

X = df.drop(["price"], axis=1)
y = df['price']
seed = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

print('Training Regression')
poly_reg_5 = PolynomialFeatures(degree = 5)
X_poly_5 = poly_reg_5.fit_transform(X_train)
lin_reg_5 = linear_model.LinearRegression()
lin_reg_5.fit(X_poly_5, y_train)
joblib.dump(lin_reg_5, 'model/price_reg.joblib')


df['good'] = df.apply(GoodRating, axis=1)
df = df.drop(['points'],1)
X1 = df.drop(["good"], axis=1)
# Extract the target feature
y1 = df['good']
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.3, random_state=42)

print('Training RandomForestClassifier')
clf_rfc = RandomForestClassifier()
clf_rfc.fit(X_train1, y_train1)
joblib.dump(clf_rfc, 'model/points_clf_RandomForestClassifier.joblib')

print('Training KNeighborsClassifier')
clf_kc = KNeighborsClassifier(3, weights='uniform', p=2, metric='euclidean')
clf_kc.fit(X_train1, y_train1)
joblib.dump(clf_kc, 'model/points_clf_KNeighborsClassifier.joblib')

print('Training DecisionTreeClassifier')
clf_dtc = DecisionTreeClassifier()
clf_dtc.fit(X_train1, y_train1)
joblib.dump(clf_dtc, 'model/points_clf_DecisionTreeClassifier.joblib')


print('Training GaussianNB')
clf_gNB = GaussianNB()
clf_gNB.fit(X_train1, y_train1)
joblib.dump(clf_gNB, 'model/points_clf_GaussianNB.joblib')
