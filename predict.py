# predict.py
import joblib
import os

# Ensure the model file is in the working directory
model_path = 'california_housing_model.joblib'

# Check if the model file exists
if not os.path.exists(model_path):
    print(f"Model file not found: {model_path}")
    exit(1)

# Load the saved model
model = joblib.load(model_path)

# Fetch the California Housing dataset
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing()
X = data.data
y = data.target

# Predict on the test set
y_pred = model.predict(X)

# Print the first few predictions to verify the model works
print("Predictions on the test data:", y_pred[:5])
