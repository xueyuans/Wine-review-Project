"""
To run this app, in your terminal:
> python wine_regression_api.py
"""
import connexion
from sklearn.externals import joblib

# Instantiate our Flask app object
app = connexion.FlaskApp(__name__, port=8080, specification_dir='swagger/')
application = app.app

# Load our pre-trained model
regr = joblib.load('./model/wine-review.joblib')

# Implement a simple health check function (GET)
def health():
    # Test to make sure our service is actually healthy
    try:
        predict(87)
    except:
        return {"Message": "Service is unhealthy"}, 500

    return {"Message": "Service is OK"}

# Implement our predict function
def predict(points):
    # Accept the feature values provided as part of our POST
    # Use these as input to clf.predict()
    prediction = regr.predict(points)

    # Return the prediction as a json
    if prediction[0] < 0:
        prediction[0] = 0;
    return prediction[0]

# Read the API definition for our service from the yaml file
app.add_api("wine_regression_api.yaml")

# Start the app
if __name__ == "__main__":
    app.run()
