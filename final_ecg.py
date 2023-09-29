import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from scipy.stats import randint

# Define data directory and paths
data_dir = r"C:\Users\lalit\OneDrive\Desktop\DTL\young_adult_ecg\ECG_GSR_Emotions\Raw Data"
stimulus_desc_path = r"C:\Users\lalit\OneDrive\Desktop\DTL\young_adult_ecg\ECG_GSR_Emotions\Stimulus_Description.xlsx"

# Load Stimulus_Description.xlsx and save it as a custom CSV
def process_excel_to_csv(input_path, output_csv=None, rename_columns=None, new_columns=None):
    df = pd.read_excel(input_path)
    if rename_columns:
        df.rename(columns=rename_columns, inplace=True)
    if new_columns:
        for column, value in new_columns.items():
            df[column] = value
    if output_csv:
        df.to_csv(output_csv, index=None, header=True)
    return df

custom_stimulus_desc_csv_path = r"C:\Users\lalit\OneDrive\Desktop\DTL\young_adult_ecg\ECG_GSR_Emotions\Stimulus_Description.csv"

stimulus_desc_file = process_excel_to_csv(
    stimulus_desc_path,
    custom_stimulus_desc_csv_path
)

# Load ECG data
def load_ecg_data(data_path, arr_shape, merged_dataframe):
    raw_data_arr = []
    for filename in os.listdir(data_path):
        if filename.endswith(".dat"):
            data = np.loadtxt(os.path.join(data_path, filename), delimiter=',')
            data = data[0:arr_shape]

            filenames = filename.split('ECGdata_')[1].split('.dat')[0].lower()
            s, p, v = map(int, (filenames.split('s')[1].split('p')[0], filenames.split('p')[1].split('v')[0], filenames.split('v')[1]))

            data_row = merged_dataframe.loc[(merged_dataframe['Session ID'] == s) &
                                            (merged_dataframe['Participant Id'] == p) &
                                            (merged_dataframe['Video ID'] == v)]
            stim_row = stimulus_desc_file.loc[(stimulus_desc_file['Session ID'] == s) &
                                              (stimulus_desc_file['Video ID'] == v)]

            for index, row in data_row.iterrows():
                raw_data_arr.append([data,
                                    row['Participant Id'], row['Session ID'], row['Video ID'],
                                    row['Name'], row['Age'], row['Gender'], row['Valence level'],
                                    row['Arousal level'], row['Dominance level'], row['Happy'],
                                    row['Sad'], row['Fear'], row['Anger'], row['Neutral'],
                                    row['Disgust'], row['Surprised'], row['Familiarity Score'],
                                    row['Emotion'], row['Valence'], row['Arousal'], row['Four_Label'],
                                    stim_row['Target Emotion'].iat[0]
                                    ])
    return raw_data_arr

# Define the array shape and initialize the data array
arr_shape = 1000
raw_data_arr = []

custom_multimodal_csv_path = r"C:\Users\lalit\OneDrive\Desktop\DTL\young_adult_ecg\ECG_GSR_Emotions\Self-annotation Multimodal_Use.csv"
custom_singlemodal_csv_path = r"C:\Users\lalit\OneDrive\Desktop\DTL\young_adult_ecg\ECG_GSR_Emotions\Self-annotation Single Modal_Use.csv"

self_annotation_multimodal_file = process_excel_to_csv(
    r"C:\Users\lalit\OneDrive\Desktop\DTL\young_adult_ecg\ECG_GSR_Emotions\Self-Annotation Labels\Self-annotation Multimodal_Use.xlsx",
    custom_multimodal_csv_path,
    rename_columns={'V_Label': 'Valence', 'A_Label': 'Arousal', 'Four_Labels': 'Four_Label'},
    new_columns={'annotation': 'M'}
)

self_annotation_singlemodal_file = process_excel_to_csv(
    r"C:\Users\lalit\OneDrive\Desktop\DTL\young_adult_ecg\ECG_GSR_Emotions\Self-Annotation Labels\Self-annotation Single Modal_Use.xlsx",
    custom_singlemodal_csv_path,
    rename_columns={'Male': 'Gender', 'Session Id': 'Session ID', 'Video Id': 'Video ID'},
    new_columns={'annotation': 'S'}
)

self_annotation_frames = [self_annotation_multimodal_file, self_annotation_singlemodal_file]
merged_dataframe = pd.concat(self_annotation_frames)

# Call the load_ecg_data function for multimodal data
multimodal_ecg_data = load_ecg_data(
    data_path=os.path.join(data_dir, "Multimodal", "ECG"),
    arr_shape=arr_shape,
    merged_dataframe=merged_dataframe
)

# Call the load_ecg_data function for single-modal data
singlemodal_ecg_data = load_ecg_data(
    data_path=os.path.join(data_dir, "Single Modal", "ECG"),
    arr_shape=arr_shape,
    merged_dataframe=merged_dataframe
)

# Define column names
cols = ['Raw Data', 'Participant ID', 'Session ID', 'Video ID', 'Name', 'Age', 'Gender', 'Valence level',
        'Arousal level', 'Dominance level', 'Happy', 'Sad', 'Fear', 'Anger', 'Neutral', 'Disgust', 'Surprised',
        'Familiarity Score', 'Emotion', 'Valence', 'Arousal', 'Four Label', 'Target Emotion']

# Create dataframes from the data arrays
multimodal_ecg_df = pd.DataFrame(multimodal_ecg_data, columns=cols)
singlemodal_ecg_df = pd.DataFrame(singlemodal_ecg_data, columns=cols)

# Rename columns for consistency
multimodal_ecg_df.rename(columns={'Participant Id': 'Participant ID', 'Four_Label': 'Four Label'}, inplace=True)
singlemodal_ecg_df.rename(columns={'Participant Id': 'Participant ID', 'Four_Label': 'Four Label'}, inplace=True)

# Fill missing values and replace NaN with empty strings
multimodal_ecg_df['Familiarity Score'] = multimodal_ecg_df['Familiarity Score'].fillna('Never watched')
multimodal_ecg_df = multimodal_ecg_df.replace(np.nan, '', regex=True)

singlemodal_ecg_df['Familiarity Score'] = singlemodal_ecg_df['Familiarity Score'].fillna('Never watched')
singlemodal_ecg_df = singlemodal_ecg_df.replace(np.nan, '', regex=True)

# Combine multimodal and single-modal dataframes if needed
# merged_ecg_df = pd.concat([multimodal_ecg_df, singlemodal_ecg_df], ignore_index=True)

# Split the dataset into features (X) and labels (y)
X_multimodal = np.array(multimodal_ecg_df['Raw Data'].tolist())
y_multimodal = np.array(multimodal_ecg_df['Emotion'].tolist())

X_singlemodal = np.array(singlemodal_ecg_df['Raw Data'].tolist())
y_singlemodal = np.array(singlemodal_ecg_df['Emotion'].tolist())

# Perform data preprocessing
sc = StandardScaler()
X_multimodal = sc.fit_transform(X_multimodal)
X_singlemodal = sc.fit_transform(X_singlemodal)

# Encode labels
label_encoder = LabelEncoder()
y_encoded_multimodal = label_encoder.fit_transform(y_multimodal)
y_encoded_singlemodal = label_encoder.fit_transform(y_singlemodal)

# Split the data into training and testing sets
X_train_multimodal, X_test_multimodal, y_train_multimodal, y_test_multimodal = train_test_split(
    X_multimodal, y_encoded_multimodal, test_size=0.2, random_state=42
)

X_train_singlemodal, X_test_singlemodal, y_train_singlemodal, y_test_singlemodal = train_test_split(
    X_singlemodal, y_encoded_singlemodal, test_size=0.2, random_state=42
)

# Define parameter grid for RandomizedSearchCV
param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False],
    'max_features': ['auto', 'sqrt', 'log2', None]
}

# Create a RandomForestClassifier
rf_classifier = RandomForestClassifier(random_state=42)

# Create RandomizedSearchCV for multimodal data
rf_random_search_multimodal = RandomizedSearchCV(
    estimator=rf_classifier,
    param_distributions=param_dist,
    n_iter=100,  # Adjust the number of iterations as needed
    cv=5,  # Adjust the number of cross-validation folds as needed
    verbose=2,
    random_state=42,
    n_jobs=-1  # Use all available CPU cores for faster search
)

# Fit RandomizedSearchCV to multimodal data
rf_random_search_multimodal.fit(X_train_multimodal, y_train_multimodal)

# Get the best parameters and best estimator for multimodal data
best_params_multimodal = rf_random_search_multimodal.best_params_
best_rf_estimator_multimodal = rf_random_search_multimodal.best_estimator_

# Use the best estimator to make predictions for multimodal data
y_pred_multimodal = best_rf_estimator_multimodal.predict(X_test_multimodal)

# Calculate and print classification report for multimodal data
classification_report_multimodal = classification_report(
    y_test_multimodal, y_pred_multimodal  # Corrected variable names here
)

# Create RandomizedSearchCV for single-modal data
rf_random_search_singlemodal = RandomizedSearchCV(
    estimator=rf_classifier,
    param_distributions=param_dist,
    n_iter=100,  # Adjust the number of iterations as needed
    cv=5,  # Adjust the number of cross-validation folds as needed
    verbose=2,
    random_state=42,
    n_jobs=-1  # Use all available CPU cores for faster search
)

# Fit RandomizedSearchCV to single-modal data
rf_random_search_singlemodal.fit(X_train_singlemodal, y_train_singlemodal)

# Get the best parameters and best estimator for single-modal data
best_params_singlemodal = rf_random_search_singlemodal.best_params_
best_rf_estimator_singlemodal = rf_random_search_singlemodal.best_estimator_

# Use the best estimator to make predictions for single-modal data
y_pred_singlemodal = best_rf_estimator_singlemodal.predict(X_test_singlemodal)

# Calculate and print classification report for single-modal data
classification_report_singlemodal = classification_report(
    y_test_singlemodal, y_pred_singlemodal  # Corrected variable names here
)

print("Classification Report for Multimodal ECG Data:\n", classification_report_multimodal)
print("\nClassification Report for Single-Modal ECG Data:\n", classification_report_singlemodal)

import joblib

# Save the best Random Forest models for multimodal and single-modal data
joblib.dump(best_rf_estimator_multimodal, 'best_rf_model_multimodal.pkl')
joblib.dump(best_rf_estimator_singlemodal, 'best_rf_model_singlemodal.pkl')

print("Models saved successfully.")