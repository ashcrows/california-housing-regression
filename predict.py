import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Load the saved model
model = joblib.load('california_housing_model.joblib')

# Fetch the California Housing dataset
data = fetch_california_housing()
X = data.data
y = data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Print the first few predictions to verify the model works
print("Predictions on test data:", y_pred[:5])
