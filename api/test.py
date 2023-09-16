import numpy as np
import pandas as pd
from zenml import steps, pipeline
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import logging


# Configura el logger para mostrar mensajes en la consola
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# create a function to predict values
def load_data_to_predict(values : dict) -> pd.DataFrame:
    # load data
    df = values
    # from dict to dataframe
    df = pd.DataFrame.from_dict(df, orient='index').T
    logger.info("Transforming the dict into dataframe")
    # transform categorical values into numerical values
    # load the encoder for each categorical column
    categorical_columns = ['Gender', 'loan_purpose', 'age', 'Region']
    for col in categorical_columns:
        with open('./models/encoder'+str(col)+'.pkl', 'rb') as file:
            encoder = pickle.load(file)
            df[col] = encoder.transform(df[col])
    # load the scaler for numerical columns ['loan_amount', 'rate_of_interest']
    numerical_columns = ['loan_amount', 'rate_of_interest']
    for col in numerical_columns:
        with open('./models/scaler'+str(col)+'.pkl', 'rb') as file:
            scaler = pickle.load(file)
            df[col] = scaler.transform(df[[col]])    
    # verify that credit score is int
    df['Credit_Score'] = df['Credit_Score'].astype(int)
    # verify the type of the numerical columns
    df['loan_amount'] = df['loan_amount'].astype(int)
    df['rate_of_interest'] = df['rate_of_interest'].astype(float)
    return df

# create a function to evaluate the model
def evaluate_model(test : pd.DataFrame):
    # load model from pickle
    with open('./models/model.pkl', 'rb') as file:
        model = pickle.load(file)
    # split the data into features and target
    X_test = test
    # evaluate the model  
    y_pred = model.predict(X_test)
    logger.info("Model evaluated succesfully")
    return y_pred


inp = {'Gender' : 'Male', 'Credit_Score' : 520, 'loan_purpose' : 'p2', 'loan_amount': 1000000 , 
       'rate_of_interest': 20 ,'age' : '25-34', 'Region' : 'North'}
# PIPELINE TESTING
def pipeline_testing(val : dict):
    data = load_data_to_predict(val)
    result = evaluate_model(data)
    return result

print(pipeline_testing(inp))

