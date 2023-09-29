import numpy as np
import pandas as pd
import joblib

# Load your test dataset (replace 'your_test_dataset.csv' with your test dataset file)
test_data = pd.read_csv(r"C:\Users\lalit\OneDrive\Desktop\DTL\final\temp_test.csv")

# Convert Celsius to Fahrenheit in the test dataset
test_data['Temperature'] = (test_data['Temperature'] * 9/5) + 32

# Define the feature (X_test) for testing
X_test = test_data[['Temperature']]

# Load the trained model (replace 'your_trained_model.joblib' with the path to your trained model)
model = joblib.load('temp.joblib')

# Make predictions on the test data
y_pred = model.predict(X_test)

# Optionally, you can also predict stress levels for new temperature values
new_temperature_celsius = float(input("Enter the temperature (in Celsius) to predict stress level: "))
new_temperature_fahrenheit = (new_temperature_celsius * 9/5) + 32
predicted_stress_level = np.clip(model.predict([[new_temperature_fahrenheit]])[0], 0, 2)
predicted_stress_level = round(predicted_stress_level)

print(f"Predicted Stress Level: {predicted_stress_level}")