import requests
import json

# Define the URL of the Flask server
url = 'http://127.0.0.1:5000/predict'

# Define the input data with different models and test data
test_data = [
    {
        "model": "decision_tree",
        "input": [7.0, 3.2, 4.7, 1.4]
    },
    {
        "model": "knn",
        "input": [6.0, 3.0, 5.0, 1.5]
    },
    # Add more test cases as needed
]

# Path to the output file
output_file = 'predictions.txt'

# Function to send request and get prediction
def get_prediction(data):
    response = requests.post(url, json=data)
    if response.status_code == 200:
        return response.json()
    else:
        return {'error': response.text}

# Open the output file in write mode
with open(output_file, 'w') as f:
    for data in test_data:
        prediction = get_prediction(data)
        if 'prediction' in prediction:
            f.write(f"Model: {data['model']}\n")
            f.write(f"Prediction: {prediction['prediction']}\n")
        else:
            f.write(f"Model: {data['model']}\n")
            f.write(f"Error: {prediction['error']}\n")
        f.write("\n\n")  # Add 2 line spaces for every prediction
