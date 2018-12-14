import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

le_country3 = LabelEncoder()

def GoodRating(row):
    if row['points'] >= 92:
        val = 3
    elif row['points'] >= 86 and row['points'] < 92:
        val = 2
    else:
        val = 1
    return(val)

df = pd.read_csv('wine-reviews_data/winemag-data-130k-v2.csv', na_values="?")
parsed_df = df[['country','price','points']]
parsed_df = parsed_df.dropna()

le_country3.fit(parsed_df['country'])
parsed_df['country']= le_country3.transform(parsed_df['country'])
parsed_df['quality_level'] = parsed_df.apply(GoodRating, axis=1)
parsed_df = parsed_df.drop(['points'],1)

X1 = parsed_df.drop(["quality_level"], axis=1)
# Extract the target feature
y1 = parsed_df['quality_level']

X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.1, random_state=42)

print('Use the country and price to predict the quality of the wine')
rfc = RandomForestClassifier()
rfc.fit(X_train1, y_train1)
joblib.dump(rfc, 'model/points_clf_RandomForestClassifier_price.joblib')
