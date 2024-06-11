import numpy as np
import pickle
import os

# Load input data (replace this with actual data loading method)
# Example: Load from a CSV file
# import pandas as pd
# input_data = pd.read_csv('path_to_your_input_data.csv').values

# For this example, let's assume the input data is a NumPy array
# Example input data: (replace with actual data)
input_data = np.array([[7.0, 3.2, 4.7, 1.4]])
print("------****----------")
print(input_data)
print("------****----------")

# Directory where models are saved
models_dir = 'models'

# List of model names
model_names = [
    'logistic_regression',
    'decision_tree',
    'random_forest',
    'svm',
    'naive_bayes',
    'knn'
]

# Function to load a model from a pickle file
def load_model(model_name):
    model_path = os.path.join(models_dir, f'{model_name}.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Load models and make predictions
for model_name in model_names:
    print(f'Loading {model_name}...')
    model = load_model(model_name)
    predictions = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data) if hasattr(model, 'predict_proba') else None
    
    print(f'\n{model_name} predictions:')
    print(f'Predicted class: {predictions}')
    if prediction_proba is not None:
        print(f'Prediction probabilities: {prediction_proba}')

# Example of a function to wrap the inference process for a single data point
def predict(model_name, data):
    model = load_model(model_name)
    prediction = model.predict(data)
    prediction_proba = model.predict_proba(data) if hasattr(model, 'predict_proba') else None
    return prediction, prediction_proba

# Example usage for a single data point
data_point = np.array([[7.0, 3.2, 4.7, 1.4]])  # replace with actual data
for model_name in model_names:
    pred, pred_proba = predict(model_name, data_point)
    print(f'\n{model_name} prediction for data point {data_point}:')
    print(f'Predicted class: {pred}')
    if pred_proba is not None:
        print(f'Prediction probabilities: {pred_proba}')