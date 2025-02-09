# Telecom Customer Churn Prediction: A Machine Learning Approach

## Overview
This project aims to predict customer churn for a telecom service provider, enabling proactive retention strategies. By leveraging machine learning, the model identifies customers at risk of leaving, allowing the company to take data-driven actions to improve customer satisfaction and revenue retention.

## Features
- **Predicts churn likelihood** based on customer demographics, service subscriptions, and billing patterns.
- **Processes raw data efficiently** by handling missing values, standardizing categorical variables, and feature engineering.
- **Trains a robust predictive model** using logistic regression and evaluates performance with cross-validation.
- **Achieves a cross-validation score of 0.842 ± 0.007 and a final test AUC of 0.858**, ensuring reliable predictions.
- **Deploys as a web service** with a REST API for real-time customer churn prediction.

## Data Processing
The dataset contains:
- **Demographics**: Gender, senior citizen status, partner and dependents.
- **Service details**: Phone service, internet service, streaming services, contract type.
- **Billing information**: Monthly charges, total charges, payment method.
- **Churn status**: Indicates whether the customer has left the service.

### Preprocessing Steps
- **Data Cleaning**: Standardized column names, formatted categorical values, and addressed missing values in 'totalcharges'.
- **Feature Engineering**:
  - Separated numerical and categorical features.
  - Applied `DictVectorizer` for categorical feature transformation.
  - Retained key numerical features like tenure and charges.
- **Target Encoding**: Converted the churn variable into a binary format for machine learning compatibility.

## Model Development
- **Data Splitting**: Divided into training and testing sets.
- **Model Selection**: Logistic Regression was chosen for its interpretability and efficiency.
- **Evaluation Metrics**:
  - Cross-validation score: **0.842 ± 0.007**
  - Final test AUC: **0.858**
- **Performance Validation**: Ensured robustness through cross-validation, demonstrating consistent predictive power.

## Deployment
To operationalize the model, it was deployed as a REST API for real-time predictions.

### Deployment Steps
1. **Model Serialization**: The trained model was saved using `pickle` for future use.
2. **API Development**: Flask was used to create an API endpoint for seamless integration.
3. **Deployment Options**:
   - **Pipenv**: Ensures isolated environments for dependency management.
   - **Docker**: Packages the service for consistent deployment across systems.
   - **Cloud Deployment**: Deploys the containerized application on cloud platforms like AWS for scalability.

### Running the Application
1. **Install dependencies**:
   ```bash
   pip install pandas numpy scikit-learn flask
   ```
2. **Run the model locally**:
   ```bash
   python predict.py
   ```
3. **Deploy as a Flask API**:
   ```bash
   flask run --host=0.0.0.0 --port=9696
   ```
4. **Send API requests**:
   ```bash
   curl -X POST -H "Content-Type: application/json" -d '{"customer_data": [values]}' http://localhost:9696/predict
   ```

## Business Impact
This system empowers telecom providers to:
- **Identify at-risk customers early** and take proactive measures.
- **Enhance customer retention** through targeted interventions.
- **Automate risk assessment** to prioritize retention efforts.
- **Leverage data-driven insights** for improved customer engagement.

## Conclusion
This project successfully bridges data science, software engineering, and business strategy to create a fully functional telecom churn prediction system. By implementing machine learning and deploying the model as a web service, it offers a practical tool for real-world customer retention efforts.
