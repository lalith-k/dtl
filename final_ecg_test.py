import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from scipy.stats import randint
import joblib

# Define data directory and paths
data_dir = r"C:\Users\lalit\OneDrive\Desktop\DTL\young_adult_ecg\ECG_GSR_Emotions\Raw Data"
stimulus_desc_path = r"C:\Users\lalit\OneDrive\Desktop\DTL\young_adult_ecg\ECG_GSR_Emotions\Stimulus_Description.xlsx"

# ... (rest of your training code remains the same)

# Define a function to make predictions using the trained models
def predict_emotion(input_data, is_multimodal=True):
    # Load the trained models
    if is_multimodal:
        best_rf_model = best_rf_estimator_multimodal
    else:
        best_rf_model = best_rf_estimator_singlemodal

    # Perform data preprocessing
    input_data = sc.transform(input_data)  # Scale the input data
    predictions = best_rf_model.predict(input_data)

    # Inverse transform the predictions to get emotion labels
    predicted_emotions = label_encoder.inverse_transform(predictions)

    return predicted_emotions

# Example usage:
# Replace input_data with your actual input data
input_data = np.random.rand(1, 1000)  # Example input data with the same shape as your ECG data
is_multimodal_input = True  # Set to True if using multimodal model, False for single-modal

predicted_emotion = predict_emotion(input_data, is_multimodal_input)
print("Predicted Emotion:", predicted_emotion[0])

# Save the best Random Forest models for multimodal and single-modal data
joblib.dump(best_rf_estimator_multimodal, 'best_rf_model_multimodal.pkl')
joblib.dump(best_rf_estimator_singlemodal, 'best_rf_model_singlemodal.pkl')

print("Models saved successfully.")