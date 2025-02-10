import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Model parameters
C = 1.0  
n_splits = 5  
output_file = f'model_C={C}.bin'

# Load dataset
df = pd.read_csv('data-week-3.csv')

# Clean to lowercase and replace spaces with underscores
df.columns = df.columns.str.lower().str.replace(' ', '_')

# Get all columns with 'object' dtype
categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)
for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

# Handle numeric conversions and missing values
df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)

# Convert target variable 'churn' to binary (0/1)
df.churn = (df.churn == 'yes').astype(int)

# Split data into train and test sets (80-20 split)
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

# Define features for the model
numerical = ['tenure', 'monthlycharges', 'totalcharges']
categorical = [
    'gender', 'seniorcitizen', 'partner', 'dependents',
    'phoneservice', 'multiplelines', 'internetservice',
    'onlinesecurity', 'onlinebackup', 'deviceprotection',
    'techsupport', 'streamingtv', 'streamingmovies',
    'contract', 'paperlessbilling', 'paymentmethod',
]

# Training function
def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)
    
    return dv, model

# Prediction function
def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')
    X = dv.transform(dicts)

    y_pred = model.predict_proba(X)[:, 1]
    return y_pred

# Validation 
print(f'doing validation with C={C}')
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)
scores = []
fold = 0

# Perform k-fold cross-validation
for train_idx, val_idx in kfold.split(df_full_train):
    # Split data into training and validation sets
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]
    
    # Get target values
    y_train = df_train.churn.values
    y_val = df_val.churn.values
    
    # Train model and make predictions
    dv, model = train(df_train, y_train, C=C)
    y_pred = predict(df_val, dv, model)
    
    # Calculate and store AUC score
    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)
    print(f'auc on fold {fold} is {auc}')
    fold = fold + 1

print('validation results:')
print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))

# Train final model on full training data
print('training the final model')
dv, model = train(df_full_train, df_full_train.churn.values, C=1.0)
y_pred = predict(df_test, dv, model)
y_test = df_test.churn.values
auc = roc_auc_score(y_test, y_pred)
print(f'auc={auc}')

# Save the model and vectorizer for later use
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)
print(f'the model is saved to {output_file}')