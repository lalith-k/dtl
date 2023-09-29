import pandas as pd
import numpy as np
import joblib

# Load your test dataset
test_df = pd.read_csv(
    r"C:\Users\lalit\OneDrive\Desktop\DTL\final\heart_rate.csv")  # Replace with the path to your test dataset

# Load the saved scaler, label encoder, and trained model
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')
best_rf_classifier = joblib.load('best_rf_classifier.pkl')  # Load the trained model

while True:
    try:
        # Enter pulse rate value in the terminal
        pulse_rate = float(input("Enter Pulse Rate (or 'q' to quit): "))

        if pulse_rate == 'q':
            break

        # Preprocess the entered pulse rate
        pulse_rate_input = np.array([[pulse_rate]])
        pulse_rate_input = scaler.transform(pulse_rate_input)

        # Predict the emotion based on the entered pulse rate
        emotion_label = label_encoder.inverse_transform(best_rf_classifier.predict(pulse_rate_input))

        print(f'Predicted Emotion: {emotion_label[0]}')

    except ValueError:
        print("Invalid input. Please enter a numeric pulse rate value or 'q' to quit.")