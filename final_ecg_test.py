import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the trained models
best_rf_model_multimodal = joblib.load('best_rf_model_multimodal.pkl')
best_rf_model_singlemodal = joblib.load('best_rf_model_singlemodal.pkl')

# Load the label encoder
label_encoder = LabelEncoder()

# Define a function to make predictions
def predict_emotion(input_data, is_multimodal=True):
    # Load the StandardScaler separately for each model
    if is_multimodal:
        scaler = joblib.load('best_rf_model_multimodal.pkl')  # Load the scaler trained for multimodal data
        model = best_rf_model_multimodal
    else:
        scaler = joblib.load('best_rf_model_singlemodal.pkl')  # Load the scaler trained for single-modal data
        model = best_rf_model_singlemodal

    # Scale the input data
    input_data = scaler.transform(input_data)

    # Make predictions
    predictions = model.predict(input_data)

    # Inverse transform the predictions to get emotion labels
    predicted_emotions = label_encoder.inverse_transform(predictions)

    return predicted_emotions

# Example usage:
# Replace `input_data` with your actual input data

input_data = np.random.rand(1, 1000)  # Example input data with the same shape as your ECG data
is_multimodal_input = True  # Set to True if using multimodal model, False for single-modal

predicted_emotion = predict_emotion(input_data, is_multimodal_input)
print("Predicted Emotion:", predicted_emotion[0])
