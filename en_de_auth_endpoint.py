from flask import Flask, request, jsonify
import pickle
import os
import numpy as np
import sqlite3
from cryptography.fernet import Fernet

app = Flask(__name__)

# Path to the models directory
models_dir = 'models/'

# Load models
def load_models(models_dir):
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    models = {}
    for model_file in model_files:
        with open(os.path.join(models_dir, model_file), 'rb') as f:
            model_name = os.path.splitext(model_file)[0]  # Use the filename without extension as key
            models[model_name] = pickle.load(f)
    return models

models = load_models(models_dir)

# Load the encryption key from the database
def get_encryption_key():
    conn = sqlite3.connect('token-creation/tokens.db')
    cursor = conn.cursor()
    cursor.execute('SELECT key FROM encryption_key WHERE id = 1')
    key = cursor.fetchone()[0]
    conn.close()
    return key

encryption_key = get_encryption_key()
cipher_suite = Fernet(encryption_key)

# Function to validate token
def validate_token(token):
    conn = sqlite3.connect('token-creation/tokens.db')
    cursor = conn.cursor()
    cursor.execute('SELECT token FROM tokens')
    tokens = cursor.fetchall()
    conn.close()

    for encrypted_token in tokens:
        try:
            decrypted_token = cipher_suite.decrypt(encrypted_token[0].encode()).decode()
            if decrypted_token == token:
                return True
        except:
            continue
    return False

# Middleware for token authentication
@app.before_request
def require_authentication():
    if request.endpoint == 'predict':  # Only apply to /predict endpoint
        token = request.args.get('Authorization')
        if not token or not validate_token(token):
            return jsonify({'error': 'Unauthorized'}), 401

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_data = data.get('input')
    
    if not input_data:
        return jsonify({'error': 'Input data is required'}), 400

    input_data = np.array(input_data).reshape(1, -1)  # Convert 1D array to 2D array

    predictions = {}
    for model_name, model in models.items():
        try:
            prediction = model.predict(input_data)
            prediction_value = prediction[0]  # Extract the single prediction value
            if isinstance(prediction_value, np.generic):
                prediction_value = prediction_value.item()  # Convert numpy scalar to native Python type
            predictions[model_name] = prediction_value
        except Exception as e:
            predictions[model_name] = str(e)

    return jsonify(predictions)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
