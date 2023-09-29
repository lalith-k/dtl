import numpy as np
import joblib

# Load the trained models
best_rf_model_multimodal = joblib.load('best_rf_model_multimodal.pkl')
best_rf_model_singlemodal = joblib.load('best_rf_model_singlemodal.pkl')

# Define a mapping between class labels and emotion names
emotion_mapping = {
    0: "Happy",
    1: "Sad",
    2: "Fear",
    3: "Anger",
    4: "Neutral",
    5: "Disgust",
    6: "Surprise",

}



# Define a function to make predictions using the trained models
def predict_emotion(input_data, is_multimodal=True):
    if is_multimodal:
        best_rf_model = best_rf_model_multimodal
    else:
        best_rf_model = best_rf_model_singlemodal

    # Perform any necessary data preprocessing on input_data here

    # Make predictions
    predictions = best_rf_model.predict(input_data)

    # Map class labels to emotion names
    predicted_emotions = [emotion_mapping[label] for label in predictions]

    return predicted_emotions


# Example usage:
# Replace input_data with your actual input data
input_data = np.random.rand(1, 1000)  # Example input data with the same shape as your ECG data
is_multimodal_input = True  # Set to True if using multimodal model, False for single-modal

predicted_emotion = predict_emotion(input_data, is_multimodal_input)
print("Predicted Emotion:", predicted_emotion)

# You can post-process the predictions or perform further actions based on your application needs
