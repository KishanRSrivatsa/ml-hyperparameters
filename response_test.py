import requests
import json

# Define the test data
test_data = [
    [5.1, 3.5, 1.4, 0.2],
    [7.0, 3.2, 4.7, 1.4],
    [6.3, 3.3, 6.0, 2.5]
    # Add more test data here
]

# URL of the prediction API
url = "http://127.0.0.1:9328/predict"

# Function to send data to the API and get the prediction
def get_prediction(data):
    response = requests.post(url, json={'features': data})
    try:
        response_data = response.json()
    except json.JSONDecodeError:
        print(f"Failed to decode JSON response: {response.text}")
        return None
    
    if 'prediction' in response_data:
        return response_data
    else:
        print(f"Unexpected response structure: {response_data}")
        return None

# List to store all predictions
predictions = []

# Iterate over the test data and get predictions
for data in test_data:
    prediction = get_prediction(data)
    if prediction is not None:
        predictions.append(prediction)
    else:
        predictions.append({'prediction': 'error'})

# Print all predictions
for i, prediction in enumerate(predictions):
    print(f"Test data {i + 1}: {test_data[i]} => Prediction: {prediction.get('prediction', 'error')}")
