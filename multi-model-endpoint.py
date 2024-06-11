from flask import Flask, request, jsonify
import pickle
import os
import numpy as np

app = Flask(__name__)

# Path to the models directory
models_dir = 'models/'

# Load models into a dictionary for easy access by name
def load_models(models_dir):
    models = {}
    for model_file in os.listdir(models_dir):
        if model_file.endswith('.pkl'):
            model_name = model_file.split('.')[0]
            with open(os.path.join(models_dir, model_file), 'rb') as f:
                models[model_name] = pickle.load(f)
    return models

models = load_models(models_dir)

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    model_name = data.get('model')
    input_data = np.array(data.get('input'))
    
    if model_name not in models:
        return jsonify({'error': 'Model not found'}), 404
    
    model = models[model_name]
    predictions = model.predict(input_data).tolist()  # Convert to list for JSON serialization
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
