import streamlit as st
import numpy as np
import joblib

# Set the page title and icon
st.set_page_config(
    page_title="Emotion and Stress Prediction",
    page_icon="😃",
)

# Add custom CSS to set the background image and style
background_image = r"C:\Users\lalit\OneDrive\Desktop\DTL\final\home_bg.jpeg"  # Replace with the path to your background image

# Define the CSS style for the background
background_style = f"""
    <style>
    .stApp {{
        background-image: url({background_image});
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
"""

# Set the background style using st.markdown
st.markdown(background_style, unsafe_allow_html=True)

# Title and description for the home page
st.title("Emotion and Stress Prediction App")
st.write("This app predicts emotions from ECG data, heart rate data, and stress levels from temperature data.")

# Sidebar options with icons
selected_option = st.sidebar.radio(
    "Select an Option",
    ["🏠 Home", "😄 Emotion Prediction from ECG Data", "❤️ Emotion Prediction from Heart Rate Data", "🌡️ Stress Prediction from Temperature Data"]
)

# Add content to the HOME page
if selected_option == "🏠 Home":
    st.header("Welcome to the Emotion and Stress Prediction App!")
    st.write("This app allows you to predict emotions and stress levels using different types of data. Use the sidebar to navigate to specific prediction sections.")

    st.subheader("Instructions:")
    st.write("1. Choose an option from the sidebar to make predictions.")
    st.write("2. Enter the required data or values in the selected section.")
    st.write("3. Click the 'Predict' button to see the results.")
    st.write("4. Animations will be displayed for stress predictions.")

# Load the trained models and functions (as shown in your previous code)
best_rf_model_multimodal_ecg = joblib.load('best_rf_model_multimodal.pkl')
best_rf_model_singlemodal_ecg = joblib.load('best_rf_model_singlemodal.pkl')

scaler_hr = joblib.load('scaler.pkl')
label_encoder_hr = joblib.load('label_encoder.pkl')
best_rf_classifier_hr = joblib.load('best_rf_classifier.pkl')

model_temp = joblib.load('temp.joblib')

# Define a mapping between class labels and emotion names for ECG
emotion_mapping_ecg = {
    0: "Happy",
    1: "Sad",
    2: "Fear",
    3: "Anger",
    4: "Neutral",
    5: "Disgust",
    6: "Surprise",
}

# Define a function to make predictions using the trained ECG models
def predict_emotion_ecg(input_data_ecg, is_multimodal=True):
    if is_multimodal:
        best_rf_model = best_rf_model_multimodal_ecg
    else:
        best_rf_model = best_rf_model_singlemodal_ecg

    # Perform any necessary data preprocessing on input_data_ecg here

    # Make predictions for ECG
    predictions_ecg = best_rf_model.predict(input_data_ecg)

    # Map class labels to emotion names for ECG
    predicted_emotions_ecg = [emotion_mapping_ecg[label] for label in predictions_ecg]

    return predicted_emotions_ecg

# Define a function to predict emotion based on heart rate
def predict_emotion_heart_rate(pulse_rate_input):
    # Preprocess the entered pulse rate
    pulse_rate_input = np.array([[pulse_rate_input]])
    pulse_rate_input = scaler_hr.transform(pulse_rate_input)

    # Predict the emotion based on the entered pulse rate for ECG
    emotion_label_hr = label_encoder_hr.inverse_transform(best_rf_classifier_hr.predict(pulse_rate_input))

    return emotion_label_hr[0]

# Define a function to predict stress levels from temperature
def predict_stress_level(temperature_input):
    # Optionally, convert Celsius to Fahrenheit
    temperature_input = (temperature_input * 9/5) + 32

    # Make predictions using the trained linear regression model for temperature
    predicted_stress_level = np.clip(model_temp.predict(np.array(temperature_input).reshape(-1, 1)), 0, 2)
    predicted_stress_level = round(predicted_stress_level[0])

    return predicted_stress_level

# Define a function to display custom animations based on stress levels
def display_stress_animation(predicted_stress_level):
    if predicted_stress_level == 0:
        st.markdown('<p style="font-size:20px;">😊 Low Stress</p>', unsafe_allow_html=True)
        st.image("low_stress.gif", use_column_width=True)
    elif predicted_stress_level == 1:
        st.markdown('<p style="font-size:20px;">😐 Moderate Stress</p>', unsafe_allow_html=True)
        st.image("moderate_stress.gif", use_column_width=True)
    elif predicted_stress_level == 2:
        st.markdown('<p style="font-size:20px;">😡 High Stress</p>', unsafe_allow_html=True)
        st.image("high_stress.gif", use_column_width=True)

# Main content based on the selected option
if selected_option == "Home":
    pass  # This is the home page, so nothing special to display here

elif selected_option == "😄 Emotion Prediction from ECG Data":
    st.subheader("Emotion Prediction from ECG Data")
    st.write("This section predicts emotions from ECG data.")

    # Generate random ECG data (example input)
    random_ecg_data = np.random.rand(1, 1000)  # Replace with your actual data
    input_data_ecg = st.text_input("Enter ECG Data (comma-separated values):", ','.join(map(str, random_ecg_data[0])))

    if st.button("Predict Emotion from ECG"):
        try:
            # Split the input string by commas and remove leading/trailing spaces
            input_data_ecg = input_data_ecg.strip().split(',')

            # Convert input data to floats
            input_data_ecg = [float(value) for value in input_data_ecg]

            # Convert to a numpy array and reshape to 2D
            input_data_ecg = np.array(input_data_ecg).reshape(1, -1)

            is_multimodal_input_ecg = True  # Set to True if using multimodal model, False for single-modal
            predicted_emotion_ecg = predict_emotion_ecg(input_data_ecg, is_multimodal_input_ecg)
            st.subheader("Predicted Emotion from ECG:")
            st.write(predicted_emotion_ecg[0])

        except ValueError:
            st.error("Invalid input format. Please enter comma-separated numeric values.")

elif selected_option == "❤️ Emotion Prediction from Heart Rate Data":
    st.subheader("Emotion Prediction from Heart Rate Data")
    st.write("This section predicts emotions from heart rate data.")

    # Enter pulse rate value
    pulse_rate = st.number_input("Enter Pulse Rate:", min_value=0.0, value=70.0)

    if st.button("Predict Emotion from Heart Rate"):
        try:
            # Predict the emotion based on the entered pulse rate for heart rate
            emotion_label_heart_rate = predict_emotion_heart_rate(pulse_rate)

            st.subheader("Predicted Emotion from Heart Rate:")
            st.write(emotion_label_heart_rate)

        except ValueError:
            st.error("Invalid input. Please enter a numeric pulse rate value.")

elif selected_option == "🌡️ Stress Prediction from Temperature Data":
    st.subheader("Stress Prediction from Temperature Data")
    st.write("This section predicts stress levels from temperature data.")

    # Enter temperature in Celsius
    temperature_celsius = st.number_input("Enter Temperature (in Celsius):", min_value=-20.0, max_value=40.0, value=25.0)

    if st.button("Predict Stress from Temperature"):
        try:
            # Predict stress levels based on the entered temperature
            predicted_stress_level = predict_stress_level(temperature_celsius)

            st.subheader("Predicted Stress Level:")
            st.write(predicted_stress_level)

            # Display animations based on stress levels
            display_stress_animation(predicted_stress_level)

        except ValueError:
            st.error("Invalid input. Please enter a numeric temperature value.")