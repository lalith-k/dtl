import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib

# Load your dataset (replace 'your_dataset.csv' with your dataset file)
data = pd.read_csv(r"C:\Users\lalit\OneDrive\Desktop\DTL\final\temp.csv")
# Convert Fahrenheit to Celsius and rescale to the range 95 to 101
data['Temperature'] = (data['Temperature'] - 32) * 5/9  # Convert to Celsius
min_temp = data['Temperature'].min()
max_temp = data['Temperature'].max()
data['Temperature'] = (data['Temperature'] - min_temp) / (max_temp - min_temp) * 6 + 95

# Define the feature (X) and target (y)
X = data[['Temperature']]
y = data['Stress Level']

# Split the data into training and testing sets (e.g., 80% training, 20% testing)
split_ratio = 0.8  # Adjust the ratio as needed
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - split_ratio, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Save the trained model to a file using joblib
joblib.dump(model, 'temp.joblib')

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Convert regression predictions to discrete stress levels (0 to 2)
predicted_stress_levels = np.clip(y_pred, 0, 2)

# Round the predicted stress levels to the nearest integer
predicted_stress_levels = np.round(predicted_stress_levels)

# Calculate accuracy on the testing data
accuracy = accuracy_score(y_test, predicted_stress_levels)

print(f"Accuracy: {accuracy:.2f}")

# Optionally, you can also predict stress levels for new temperature values
new_temperature = float(input("Enter the temperature (in Celsius) to predict stress level: "))
# Rescale the new temperature value to match the range of 95 to 101
new_temperature = (new_temperature - min_temp) / (max_temp - min_temp) * 6 + 95
predicted_stress_level = np.clip(model.predict([[new_temperature]])[0], 0, 2)
predicted_stress_level = round(predicted_stress_level)

print(f"Predicted Stress Level: {predicted_stress_level}")