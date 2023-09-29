import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

# Load your dataset
df = pd.read_csv(r"C:\Users\lalit\OneDrive\Desktop\DTL\final\heart_rate_emotion_dataset.csv\heart_rate_emotion_dataset.csv")

# Assuming your CSV file has columns 'HeartRate' for features and 'Emotion' for labels
X_features = df['HeartRate'].values
y_labels = df['Emotion'].values

# Preprocess the data
scaler = StandardScaler()
X_features = scaler.fit_transform(X_features.reshape(-1, 1))  # Reshape to 2D array

label_encoder = LabelEncoder()
y_labels = label_encoder.fit_transform(y_labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=0.2, random_state=42)

# Define an expanded hyperparameter grid to search over
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy']
}

# Create a Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42, class_weight='balanced')

# Perform GridSearchCV for hyperparameter tuning with more iterations
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid,
                           scoring='accuracy', cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best hyperparameters from the grid search
best_params = grid_search.best_params_

# Train the model with the best hyperparameters
best_rf_classifier = RandomForestClassifier(random_state=42, class_weight='balanced', **best_params)
best_rf_classifier.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = best_rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')