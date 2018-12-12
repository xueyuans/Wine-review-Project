"""
To run this app, in your terminal:
> python wine_classification_description_api.py
"""
import connexion
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from train_model_description import vectorizer
import numpy

# Instantiate our Flask app object
app = connexion.FlaskApp(__name__, port=8080, specification_dir='swagger/')
application = app.app

# Load our pre-trained model
clf_rfc = joblib.load('./model/points_clf_RandomForestClassifier_description.joblib')

# Implement a simple health check function (GET)
def health():
    # Test to make sure our service is actually healthy
    try:
        clf_predict("Aromas include tropical fruit, broom, brimstone and dried herb. The palate isn't overly expressive, offering unripened apple, citrus and dried sage alongside brisk acidity")
    except:
        return {"Message": "Service is unhealthy"}, 500

    return {"Message": "Service is OK"}

# Implement our predict function
def clf_predict(description):
    # Accept the feature values provided as part of our POST
    # Use these as input to clf.predict()
    description = [description]
    description = vectorizer.transform(description)
    prediction = clf_rfc.predict(description)

    if prediction[0] == 3:
        predicted_class = "excellent"
    elif prediction[0] == 2:
        predicted_class = "good"
    else:
        predicted_class = "not good"

    # Return the prediction as a json
    return {"prediction" : predicted_class}



# Read the API definition for our service from the yaml file
app.add_api("wine_classification_description_api.yaml")

# Start the app
if __name__ == "__main__":
    app.run()
