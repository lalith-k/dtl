import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Set the page title and icon
st.set_page_config(
    page_title="Emotion and Stress Prediction",
    page_icon="😃",
)

# Title and description
st.title("Emotion and Stress Prediction App")
st.write("This app predicts emotions from ECG and heart rate data and stress levels from temperature data.")

# Sidebar options
selected_option = st.sidebar.selectbox("Select an Option", ["Emotion Prediction from ECG Data", "Emotion Prediction from Heart Rate Data", "Stress Prediction from Temperature Data"])

# Load the label encoders and scalers
label_encoder_ecg = joblib.load('label_encoder.pkl')
label_encoder_hr = joblib.load('label_encoder_hr.pkl')
scaler_ecg = joblib.load('scaler.pkl')
scaler_hr = joblib.load('scaler_hr.pkl')

# Load the trained models
loaded_model_ecg = joblib.load('best_rf_model_multimodal.pkl')
loaded_model_hr = joblib.load('best_rf_classifier.pkl')

# Define a function to predict emotions from ECG data
def predict_emotion_ecg(input_data):
    # Scale the input data
    input_data = scaler_ecg.transform(input_data)

    # Make predictions
    predictions = loaded_model_ecg.predict(input_data)

    # Inverse transform the predictions to get emotion labels
    predicted_emotions = label_encoder_ecg.inverse_transform(predictions)

    return predicted_emotions[0]

# Define a function to predict emotions from heart rate
def predict_emotion_heart_rate(heart_rate_input):
    # Scale the input heart rate
    heart_rate_input = scaler_hr.transform(np.array(heart_rate_input).reshape(-1, 1))

    # Make predictions
    predicted_label = loaded_model_hr.predict(heart_rate_input)

    # Inverse transform the predicted label to get the emotion
    predicted_emotion = label_encoder_hr.inverse_transform(predicted_label)

    return predicted_emotion[0]

# Define a function to predict stress levels from temperature
def predict_stress_level(temperature_input):
    # Optionally, convert Celsius to Fahrenheit
    temperature_input = (temperature_input * 9/5) + 32

    # Make predictions using the trained linear regression model
    predicted_stress_level = np.clip(model.predict(np.array(temperature_input).reshape(-1, 1)), 0, 2)
    predicted_stress_level = round(predicted_stress_level)

    return predicted_stress_level

# Main content based on the selected option
if selected_option == "Emotion Prediction from ECG Data":
    st.subheader("Emotion Prediction from ECG Data")
    st.write("This section predicts emotions from ECG data.")

    # Enter ECG data (example input)
    input_data_ecg = st.text_input("Enter ECG Data (comma-separated values):")
    if st.button("Predict Emotion"):
        try:
            input_data_ecg = np.array([list(map(float, input_data_ecg.split(',')))])
            predicted_emotion_ecg = predict_emotion_ecg(input_data_ecg)
            st.subheader("Predicted Emotion:")
            st.write(predicted_emotion_ecg)
        except ValueError:
            st.error("Invalid input format. Please enter comma-separated numeric values.")

elif selected_option == "Emotion Prediction from Heart Rate Data":
    st.subheader("Emotion Prediction from Heart Rate Data")
    st.write("This section predicts emotions from heart rate data.")

    # Enter pulse rate value
    pulse_rate = st.number_input("Enter Pulse Rate:", min_value=0.0, value=70.0)

    if st.button("Predict Emotion"):
        try:
            # Preprocess the entered pulse rate
            pulse_rate_input = np.array([[pulse_rate]])
            pulse_rate_input = scaler_hr.transform(pulse_rate_input)

            # Predict the emotion based on the entered pulse rate
            emotion_label = label_encoder_hr.inverse_transform(loaded_model_hr.predict(pulse_rate_input))

            # Display the predicted emotion
            st.subheader("Predicted Emotion:")
            st.write(emotion_label[0])
        except ValueError:
            st.error("Invalid input. Please enter a numeric pulse rate value.")

elif selected_option == "Stress Prediction from Temperature Data":
    st.subheader("Stress Prediction from Temperature Data")
    st.write("This section predicts stress levels from temperature data.")

    # Enter temperature in Celsius
    temperature_celsius = st.number_input("Enter Temperature (in Celsius):", min_value=-20.0, max_value=40.0, value=25.0)

    if st.button("Predict Stress Level"):
        try:
            # Convert Celsius to Fahrenheit
            temperature_fahrenheit = (temperature_celsius * 9/5) + 32

            # Make predictions using the trained model
            predicted_stress_level = predict_stress_level(temperature_fahrenheit)

            # Display the predicted stress level
            st.subheader("Predicted Stress Level:")
            st.write(predicted_stress_level)
        except ValueError:
            st.error("Invalid input. Please enter a numeric temperature value in Celsius.")

# Optionally, you can add a section for new temperature predictions
# ...

