import numpy as np
import pandas as pd
from zenml import steps, pipeline
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from ..utilities.functions import phik_matriz
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle


# create a function to predict values
def load_data_to_predict() -> pd.DataFrame:
    # load data
    df = pd.read_csv('./data/Loan_Default.csv')
    print("Data loaded succesfully") 
    # delete the missing values
    df = df.dropna()
    print("Missing values deleted")
    # filter the useful columns
    useful_columns =  phik_matriz(df)
    df_f = df[useful_columns]
    return df_f

# create a function to evaluate the model
def evaluate_model(model, test : pd.DataFrame):
    # load model from pickle
    with open('../../models/model.pkl', 'rb') as file:
        model = pickle.load(file)
    # split the data into features and target
    X_test = test.drop(['Status'], axis = 1)
    y_test = test['Status']
    # evaluate the model  
    y_pred = model.predict(X_test)
    metrics = {'accuracy': accuracy_score(y_test, y_pred),
               'precision': precision_score(y_test, y_pred),
               'recall' : recall_score(y_test, y_pred),
               'f1': f1_score(y_test, y_pred)}
    # save the metrics as json
    import json
    with open('../metrics/metrics.json', 'w') as file:
        json.dump(metrics, file)