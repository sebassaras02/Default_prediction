import numpy as np
import pandas as pd
from typing import Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from functions import read_data, save_file_blob_azure
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pickle
import logging
import os


# Configura el logger para mostrar mensajes en la consola
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#enter credentials
account_name = os.environ.get('SECRET_NAME_BS_AZURE')  
account_key = os.environ.get('SECRET_KEY_BS_AZURE')  

# STEP 1 OF PIPELINE
# create a function to process data
def load_data_to_train() -> Tuple[pd.DataFrame, pd.DataFrame]:
    # load data
    df = read_data(account_name, account_key, 'datos', 'Loan_Default.csv')
    logger.info("Data loaded succesfully") 
    # inpute categorical columns with the mode of the dataframe
    categorical_columns = ['Gender', 'Credit_Score', 'loan_purpose', 'age', 'Status', 'Region']
    # fill with the mode the NaN values
    for col in categorical_columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    # fill numeric columns with the mean
    numerical_columns = ['loan_amount', 'rate_of_interest']
    for col in numerical_columns:
        df[col] = df[col].fillna(df[col].mean())
    # get the important columns for the algorithm
    df_f = df[['Gender', 'Credit_Score', 'loan_purpose', 'loan_amount', 'rate_of_interest','age', 'Status', 'Region']]
    # get categorical columns of the dataframe
    categorical_columns = df_f.select_dtypes(include = ['object']).columns.tolist()
    for col in categorical_columns:
        encoder = LabelEncoder()
        df_f[col] = encoder.fit_transform(df_f[col])  
        with open('../models/encoder'+str(col)+'.pkl', 'wb') as file:
            pickle.dump(encoder, file)
    # apply a min-max scaler to the numerical columns
    for col in numerical_columns:
        scaler = MinMaxScaler()
        df_f[col] = scaler.fit_transform(df_f[[col]])  
        with open('../models/scaler'+str(col)+'.pkl', 'wb') as file:
            pickle.dump(scaler, file)
    # split data into train and test
    train, test = train_test_split(df_f, test_size = 0.3, random_state = 99)
    logger.info("Data processed succesfully")
    return train, test


# STEP 2 OF PIPELINE
# create a function to train the model
def train_model(train : pd.DataFrame, test : pd.DataFrame):
    # split the data into features and target
    y_train = train['Status']
    X_train = train.drop(['Status'], axis = 1)
    y_test = test['Status']
    X_test = test.drop(['Status'], axis = 1)
    # train the model
    model = LGBMClassifier(n_estimators = 100, random_state = 99)
    model.fit(X_train, y_train)
    logger.info("Model trained succesfully")
    # save model as pickle
    with open('../models/model.pkl', 'wb') as file:
        pickle.dump(model, file)
        logging.debug("Model saved succesfully")
    # save the metrics as json file
    import json
    y_pred = model.predict(X_test)
    metrics = {'accuracy': accuracy_score(y_test, y_pred),
               'precision': precision_score(y_test, y_pred, pos_label = 1),
               'recall' : recall_score(y_test, y_pred, pos_label = 1),
               'f1': f1_score(y_test, y_pred, pos_label = 1)}
    with open('../metrics/metrics_retraining.json', 'w') as file:
        json.dump(metrics, file)
    logger.info("Metrics saved succesfully")
        
# CREATE PIPELINE
def pipeline_training(name : str = 'Loan Default Training'):
    train, test = load_data_to_train()
    train_model(train, test)
    