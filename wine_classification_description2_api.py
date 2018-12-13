"""
To run this app, in your terminal:
> python wine_classification_description2.py
"""
import connexion
from train_model_description2 import le_country, le_taster_name
from sklearn.externals import joblib
from sklearn.preprocessing import PolynomialFeatures
import numpy

# Instantiate our Flask app object
app = connexion.FlaskApp(__name__, port=8080, specification_dir='swagger/')
application = app.app

# Load our pre-trained model
clf_rfc = joblib.load('./model/points_clf_RandomForestClassifier2.joblib')
clf_kc = joblib.load('./model/points_clf_KNeighborsClassifier2.joblib')
clf_dtc = joblib.load('./model/points_clf_DecisionTreeClassifier2.joblib')
clf_gNB = joblib.load('./model/points_clf_GaussianNB2.joblib')

# Implement a simple health check function (GET)
def health():
    # Test to make sure our service is actually healthy
    try:
        clf_predict('RandomForestClassifier', 'US', 'Aromas include tropical fruit, broom, brimstone and dried herb',14, 'Paul Gregutt')
    except:
        return {"Message": "Service is unhealthy"}, 500

    return {"Message": "Service is OK"}

# Implement our predict function
def clf_predict(model, country, description, price, taster_name):
    # Accept the feature values provided as part of our POST
    # Use these as input to clf.predict()
    countrys = []
    countrys.append(country)
    country = le_country.transform(countrys)

    description_length = len(description)

    taster_names = []
    taster_names.append(taster_name)
    taster_name = le_taster_name.transform(taster_names)

    row_data = [[]]
    row_data[0].append(country[0])
    row_data[0].append(price)
    row_data[0].append(taster_name[0])
    row_data[0].append(description_length)

    prediction = []
    if model == "RandomForestClassifier":
        prediction = clf_rfc.predict(row_data);
    elif  model == "KNeighborsClassifier":
        prediction = clf_kc.predict(row_data);
    elif model == "DecisionTreeClassifier":
        prediction = clf_dtc.predict(row_data);
    else:
        prediction = clf_gNB.predict(row_data);

    if prediction[0] == 3:
        predicted_class = "excellent"
    elif prediction[0] == 2:
        predicted_class = "good"
    else:
        predicted_class = "not good"

    # Return the prediction as a json
    return {"prediction" : predicted_class}



# Read the API definition for our service from the yaml file
app.add_api("wine_classification_description2_api.yaml")

# Start the app
if __name__ == "__main__":
    app.run()
