import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo


def create_model():
    # Fetch the wine quality dataset
    wine_quality = fetch_ucirepo(id=186)

    # Select the 11 features and the target variable
    X = wine_quality.data.features
    y = wine_quality.data.targets

    # Normalize features using StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Define the model (MLP Regressor)
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32), activation="relu", solver="adam", max_iter=1000
    )

    # Fit the model
    model.fit(X, y)

    return model
