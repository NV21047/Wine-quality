from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
from ucimlrepo import fetch_ucirepo
from sklearn.neural_network import MLPRegressor

app = Flask(__name__)

# Fetch the wine quality dataset
wine_quality = fetch_ucirepo(id=186)
X = wine_quality.data.features
y = wine_quality.data.targets

# Instantiate MLPRegressor
trained_model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000)

# Assuming y is a pandas DataFrame
y_array = np.array(y)

# Flatten the target variable y
y_flattened = y_array.ravel()

# Train the model
trained_model.fit(X, y_flattened)

# Save the trained model using joblib
joblib.dump(trained_model, "model.joblib")

# Define the variables used in the form
variables = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]


@app.route("/")
def index():
    return render_template("index.html", variables=variables)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json  # Get data from the JSON request

    # Use the loaded model to make predictions
    model = joblib.load("model.joblib")
    features = np.array(list(data.values())).reshape(1, -1)
    quality_prediction = model.predict(features)

    # Return the prediction as JSON
    return jsonify({"quality": round(float(quality_prediction[0]), 2)})


@app.route("/result")
def result():
    quality = request.args.get("quality")
    return render_template("result.html", quality=quality)


if __name__ == "__main__":
    app.run(debug=True)
