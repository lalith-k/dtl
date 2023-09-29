import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the trained Random Forest classifier
best_rf_classifier = joblib.load('best_rf_classifier.pkl')

# Load the label encoder and standard scaler used during training
label_encoder = joblib.load('label_encoder.pkl')
scaler = joblib.load('scaler.pkl')

# Define a function to make predictions
def predict_emotion(heart_rate):
    # Scale the input heart rate using the same scaler used during training
    heart_rate = scaler.transform(np.array(heart_rate).reshape(-1, 1))

    # Make predictions
    predicted_label = best_rf_classifier.predict(heart_rate)

    # Inverse transform the predicted label to get the emotion
    predicted_emotion = label_encoder.inverse_transform(predicted_label)

    return predicted_emotion[0]

# Example usage:
# Replace `heart_rate_input` with your actual heart rate input
heart_rate_input = [70]  # Example heart rate input

predicted_emotion = predict_emotion(heart_rate_input)
print("Predicted Emotion:", predicted_emotion)