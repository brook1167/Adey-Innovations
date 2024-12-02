from flask import Flask, request, jsonify
import joblib
import numpy as np
import logging

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load("./mlruns/model.pkl")  # Path to your saved model



# Define API endpoints
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array(data['features']).reshape(1, -1)  # Adjust reshape to match model's expected input shape
    prediction = model.predict(features)
    
    
    return jsonify({'prediction': int(prediction[0])})

@app.errorhandler(Exception)
def handle_exception(e):
    
    return jsonify(error=str(e)), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)