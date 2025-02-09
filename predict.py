"""
Telecom Customer Churn Prediction Service
---------------------------------------

This script implements a Flask web service for predicting customer churn in a telecommunications company.
It loads a pre-trained machine learning model and provides an API endpoint for making real-time predictions.

Key Components:
--------------
- Model Loading: Loads serialized model and vectorizer from disk
- Flask Service: Creates REST API endpoint for predictions
- JSON Processing: Handles customer data in JSON format
- Prediction Logic: Transforms data and generates churn probabilities

API Endpoint:
------------
POST /predict
   Accepts customer data in JSON format and returns churn predictions

Input Format:
------------
{
   "gender": str,
   "seniorcitizen": int,
   "partner": str,
   "dependents": str,
   "phoneservice": str,
   "multiplelines": str,
   "internetservice": str,
   "onlinesecurity": str,
   "onlinebackup": str,
   "deviceprotection": str,
   "techsupport": str,
   "streamingtv": str,
   "streamingmovies": str,
   "contract": str,
   "paperlessbilling": str,
   "paymentmethod": str,
   "tenure": int,
   "monthlycharges": float,
   "totalcharges": float
}

Output Format:
-------------
{
   "churn_probability": float,
   "churn": bool
}

Usage:
------
1. Ensure model file 'model_C=1.0.bin' is in the same directory
2. Run the script: python predict.py
3. Service will be available at http://localhost:9696/predict

Dependencies:
------------
- Flask
- pickle
- scikit-learn (implied through model loading)

"""

import pickle
from flask import Flask, request, jsonify

# Load the saved model and vectorizer
model_file = 'model_C=1.0.bin'
with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('churn')

@app.route('/predict', methods=['POST'])
def predict():
    # Get customer data from POST request
    customer = request.get_json()
    
    # Make prediction
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5
    
    # Format response
    result = {
        'churn_probability': float(y_pred),
        'churn': bool(churn)
    }
    
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)