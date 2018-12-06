"""
To run this app, in your terminal:
> python wine_regression_api.py
"""
import connexion
from train_model import le_country, le_province, le_taster_name
from sklearn.externals import joblib
from sklearn.preprocessing import PolynomialFeatures
import numpy

# Instantiate our Flask app object
app = connexion.FlaskApp(__name__, port=8080, specification_dir='swagger/')
application = app.app

# Load our pre-trained model
regr = joblib.load('./model/price_reg.joblib')

# Implement a simple health check function (GET)
def health():
    # Test to make sure our service is actually healthy
    try:
        reg_predict('US', 'Oregon', 87, 'Paul Gregutt')
        clf_predict('US','Oregon',14, 'Paul Gregutt')
    except:
        return {"Message": "Service is unhealthy"}, 500

    return {"Message": "Service is OK"}

# Implement our predict function
def reg_predict(country, province, points, taster_name):
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
    row_data[0].append(points)
    row_data[0].append(province[0])
    row_data[0].append(taster_name[0])
    poly_reg_5 = PolynomialFeatures(degree = 5)
    X_poly_5 = poly_reg_5.fit_transform(row_data)

    prediction = regr.predict(X_poly_5)

    # Return the prediction as a json
    if prediction[0] < 0:
        prediction[0] = 0;
    return prediction[0]

# Read the API definition for our service from the yaml file
app.add_api("wine_regression_api.yaml")

# Start the app
if __name__ == "__main__":
    app.run()
