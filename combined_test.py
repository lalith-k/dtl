import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd

# Load the trained models for emotion prediction from ECG data
best_rf_model_multimodal = joblib.load('best_rf_model_multimodal.pkl')
best_rf_model_singlemodal = joblib.load('best_rf_model_singlemodal.pkl')

# Load the label encoder for emotion prediction
label_encoder = LabelEncoder()

# Load the trained Random Forest classifier for emotion prediction from heart rate
best_rf_classifier = joblib.load('best_rf_classifier.pkl')

# Load the label encoder and standard scaler for heart rate prediction
heart_rate_label_encoder = joblib.load('label_encoder.pkl')
heart_rate_scaler = joblib.load('scaler.pkl')

# Load your test dataset for stress prediction (replace 'your_test_dataset.csv' with your file)
test_data = pd.read_csv('your_test_dataset.csv')

# Convert Celsius to Fahrenheit in the test dataset
test_data['Temperature'] = (test_data['Temperature'] * 9/5) + 32

# Define a function to predict emotions from ECG data
def predict_emotion_ecg(input_data, is_multimodal=True):
    # Load the StandardScaler separately for each model
    if is_multimodal:
        scaler = joblib.load('best_rf_model_multimodal_scaler.pkl')
        model = best_rf_model_multimodal
    else:
        scaler = joblib.load('best_rf_model_singlemodal_scaler.pkl')
        model = best_rf_model_singlemodal

    # Scale the input data
    input_data = scaler.transform(input_data)

    # Make predictions
    predictions = model.predict(input_data)

    # Inverse transform the predictions to get emotion labels
    predicted_emotions = label_encoder.inverse_transform(predictions)

    return predicted_emotions

# Define a function to predict emotions from heart rate
def predict_emotion_heart_rate(heart_rate_input):
    # Scale the input heart rate using the same scaler used during training
    heart_rate_input = heart_rate_scaler.transform(np.array(heart_rate_input).reshape(-1, 1))

    # Make predictions
    predicted_label = best_rf_classifier.predict(heart_rate_input)

    # Inverse transform the predicted label to get the emotion
    predicted_emotion = heart_rate_label_encoder.inverse_transform(predicted_label)

    return predicted_emotion[0]

# Define a function to predict stress levels from temperature
def predict_stress_level(temperature_input):
    # Optionally, convert Celsius to Fahrenheit
    temperature_input = (temperature_input * 9/5) + 32

    # Make predictions using the trained linear regression model
    predicted_stress_level = np.clip(model.predict(np.array(temperature_input).reshape(-1, 1)), 0, 2)
    predicted_stress_level = round(predicted_stress_level)

    return predicted_stress_level

# Example usage:
# Replace input_data_ecg, heart_rate_input, and temperature_input with your actual data
input_data_ecg = np.random.rand(1, 1000)  # Example ECG data
heart_rate_input = [70]  # Example heart rate input
temperature_input = [25.0]  # Example temperature input

# Emotion prediction from ECG data
predicted_emotion_ecg = predict_emotion_ecg(input_data_ecg, is_multimodal=True)
print("Emotion Prediction from ECG Data:", predicted_emotion_ecg[0])

# Emotion prediction from heart rate
predicted_emotion_heart_rate = predict_emotion_heart_rate(heart_rate_input)
print("Emotion Prediction from Heart Rate:", predicted_emotion_heart_rate)

# Stress prediction from temperature
predicted_stress = predict_stress_level(temperature_input)
print("Stress Prediction from Temperature:", predicted_stress)
