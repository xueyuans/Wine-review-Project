"""
To run this app, in your terminal:
> python wine_regression_api.py
"""
import connexion
from train_model import le_country, le_province, le_taster_name
from train_model_description2 import le_country2, le_taster_name2
from train_model_price import le_country3
from sklearn.externals import joblib
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_extraction.text import CountVectorizer
from train_model_description import vectorizer

# Instantiate our Flask app object
app = connexion.FlaskApp(__name__, port=8080, specification_dir='swagger/')
application = app.app

# Load our pre-trained model
clf_rfc = joblib.load('./model/points_clf_RandomForestClassifier.joblib')
clf_kc = joblib.load('./model/points_clf_KNeighborsClassifier.joblib')
clf_dtc = joblib.load('./model/points_clf_DecisionTreeClassifier.joblib')
clf_gNB = joblib.load('./model/points_clf_GaussianNB.joblib')

clf_rfc2 = joblib.load('./model/points_clf_RandomForestClassifier2.joblib')
clf_kc2 = joblib.load('./model/points_clf_KNeighborsClassifier2.joblib')
clf_dtc2 = joblib.load('./model/points_clf_DecisionTreeClassifier2.joblib')
clf_gNB2 = joblib.load('./model/points_clf_GaussianNB2.joblib')

clf_rfc3 = joblib.load('./model/points_clf_RandomForestClassifier_description.joblib')

clf_rfc4 = joblib.load('./model/points_clf_RandomForestClassifier_price.joblib')

# Implement a simple health check function (GET)
def health():
    # Test to make sure our service is actually healthy
    try:
        clf_predict1('RandomForestClassifier', 'US','Oregon',14, 'Paul Gregutt')
        clf_predict2('RandomForestClassifier', 'US',14, 'Paul Gregutt','Aromas include tropical fruit, broom, brimstone and dried herb')
        clf_predict3("Aromas include tropical fruit, broom, brimstone and dried herb.")
        clf_predict4(87)
    except:
        return {"Message": "Service is unhealthy"}, 500

    return {"Message": "Service is OK"}

# Implement our predict function
def clf_predict1(model, country, province, price, taster_name):
    # Accept the feature values provided as part of our POST
    # Use these as input to clf.predict()
    countrys = []
    countrys.append(country)
    country = le_country.transform(countrys)

    provinces = []
    provinces.append(province)
    province = le_province.transform(provinces)

    taster_names = []
    taster_names.append(taster_name)
    taster_name = le_taster_name.transform(taster_names)

    row_data = [[]]
    row_data[0].append(country[0])
    row_data[0].append(price)
    row_data[0].append(province[0])
    row_data[0].append(taster_name[0])

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

# Implement our predict function
def clf_predict2(model, country, price, taster_name, description):
    # Accept the feature values provided as part of our POST
    # Use these as input to clf.predict()
    countrys = []
    countrys.append(country)
    country = le_country2.transform(countrys)

    description_length = len(description)

    taster_names = []
    taster_names.append(taster_name)
    taster_name = le_taster_name2.transform(taster_names)

    row_data = [[]]
    row_data[0].append(country[0])
    row_data[0].append(price)
    row_data[0].append(taster_name[0])
    row_data[0].append(description_length)

    prediction = []
    if model == "RandomForestClassifier":
        prediction = clf_rfc2.predict(row_data);
    elif  model == "KNeighborsClassifier":
        prediction = clf_kc2.predict(row_data);
    elif model == "DecisionTreeClassifier":
        prediction = clf_dtc2.predict(row_data);
    else:
        prediction = clf_gNB2.predict(row_data);

    if prediction[0] == 3:
        predicted_class = "excellent"
    elif prediction[0] == 2:
        predicted_class = "good"
    else:
        predicted_class = "not good"
    # Return the prediction as a json
    return {"prediction" : predicted_class}

# Implement our predict function
def clf_predict3(description):
    # Accept the feature values provided as part of our POST
    # Use these as input to clf.predict()
    description = [description]
    description = vectorizer.transform(description)
    prediction = clf_rfc3.predict(description)

    if prediction[0] == 3:
        predicted_class = "excellent"
    elif prediction[0] == 2:
        predicted_class = "good"
    else:
        predicted_class = "not good"

    # Return the prediction as a json
    return {"prediction" : predicted_class}

# Implement our predict function
def clf_predict4(country, price):
    # Accept the feature values provided as part of our POST
    # Use these as input to clf.predict()
    countrys = []
    countrys.append(country)
    country = le_country3.transform(countrys)

    row_data = [[]]
    row_data[0].append(price)
    row_data[0].append(country)
    prediction = clf_rfc4.predict(row_data)

    if prediction[0] == 3:
        predicted_class = "excellent"
    elif prediction[0] == 2:
        predicted_class = "good"
    else:
        predicted_class = "not good"

    # Return the prediction as a json
    return {"prediction" : predicted_class}

# Read the API definition for our service from the yaml file
app.add_api("wine_classification_api.yaml")

# Start the app
if __name__ == "__main__":
    app.run()
