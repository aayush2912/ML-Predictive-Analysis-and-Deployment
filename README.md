# Customer Churn Prediction System for Telecom Services

## Overview
This project aims to predict customer churn for a telecom service provider. By analyzing historical customer data, the model identifies customers who are likely to leave (churn), allowing the company to take proactive steps to retain them and improve customer satisfaction.

## Features
- **Predicts churn likelihood** based on customer demographics, service details, and billing information.
- **Processes raw data** by cleaning text, handling missing values, and converting categorical variables into numerical format.
- **Builds a predictive model** using logistic regression and evaluates its performance with cross-validation.
- **Achieves 85.8% accuracy**, helping the company prioritize retention efforts effectively.

## Data Processing
The project uses customer data containing:
- **Demographics**: Gender, senior citizen status
- **Service details**: Phone service, internet service, streaming services
- **Billing information**: Monthly charges, total charges
- **Churn status**: Whether the customer has left the service

Data is preprocessed by:
- Cleaning text data (lowercasing, removing spaces)
- Handling missing values
- Encoding categorical features into numerical values

## Model Building
- The dataset is **split into training and testing sets**.
- A **logistic regression model** is trained to identify churn patterns.
- The model is evaluated using **cross-validation** to ensure reliability.
- Predictions are generated to rank customers by their likelihood to churn.

## Results
- The model achieves **85.8% accuracy** in predicting customer churn.
- This allows the telecom company to:
  - Identify at-risk customers
  - Implement targeted retention strategies
  - Improve overall customer experience

## Deployment
Model deployment is crucial when using the model across different machines or applications without retraining or rerunning the code. By deploying the model as a web service, external systems (like marketing services) can send requests to the server to get predictions, such as whether a customer is likely to churn. Based on the prediction, actions like sending promotional offers can be automated.

### Deployment Steps
1. **Train and Save the Model**: After training, save the model as a file for future predictions (see session 02-pickle).
2. **Create API Endpoints**: Develop API endpoints using Flask to request predictions from the model (see sessions 03-flask-intro and 04-flask-deployment).
3. **Server Deployment Options**:
   - **Pipenv**: Create isolated environments to manage Python dependencies, ensuring they don’t interfere with other services on the machine.
   - **Docker**: Package the service in a Docker container, including system and Python dependencies, for consistent deployment across environments.
   - **Cloud Deployment**: Deploy the Docker container to a cloud service like AWS for global accessibility, scalability, and reliability.

## Usage
1. **Install dependencies**:
   ```bash
   pip install pandas numpy scikit-learn flask
   ```
2. **Run the model**:
   ```bash
   python churn_prediction.py
   ```
3. **Deploy as a service**:
   ```bash
   flask run --host=0.0.0.0 --port=5000
   ```
4. **Send API requests**:
   ```bash
   curl -X POST -H "Content-Type: application/json" -d '{"customer_data": [values]}' http://localhost:5000/predict
   ```

## Conclusion
This system provides a valuable tool for telecom providers to understand and mitigate customer churn. By leveraging predictive analytics and deploying the model as a web service, businesses can enhance customer loyalty and optimize their service offerings.
