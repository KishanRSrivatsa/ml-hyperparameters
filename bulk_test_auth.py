import requests
import json

# Define the URL of the Flask server
url = 'http://localhost:5000/predict'

# Define the input data
test_data = [
    [5.1, 3.5, 1.4, 0.2],
    [4.9, 3.0, 1.4, 0.2],
    [4.7, 3.2, 1.3, 0.2],
    [4.6, 3.1, 1.5, 0.2],
    [5.0, 3.6, 1.4, 0.2],
    [5.4, 3.9, 1.7, 0.4],
    [4.6, 3.4, 1.4, 0.3]
]

# Use one of the generated tokens
token = 'your_generated_token'

# Path to the output file
output_file = 'predictions.txt'

# Function to send request and get predictions
def get_predictions(input_data):
    headers = {'Authorization': token}
    response = requests.post(url, json={'input': input_data}, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        return {'error': response.text}

# Open the output file in write mode
with open(output_file, 'w') as f:
    for data in test_data:
        f.write(f"test_data : {', '.join(map(str, data))}\n")
        predictions = get_predictions(data)
        for model_name, prediction in predictions.items():
            readable_model_name = model_name.replace("_", " ").title().replace(" ", "_").lower()
            f.write(f"Model: {readable_model_name}\n")
            f.write(f"Prediction: {prediction}\n")
        f.write("\n\n")  # Add 2 line spaces for every set of predictions
