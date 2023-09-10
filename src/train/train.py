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

# create a function to process data
def load_data_to_train() -> List:
    # load data
    df = pd.read_csv('./data/Loan_Default.csv')
    print("Data loaded succesfully") 
    # delete the missing values
    df = df.dropna()
    print("Missing values deleted")
    # filter the useful columns
    useful_columns =  phik_matriz(df)
    df_f = df[useful_columns]
    # split data into train and test
    train, test = train_test_split(df_f, test_size = 0.3, random_state = 42)
    return [train, test]

# create a function to train the model
def train_model(train : pd.DataFrame):
    # split the data into features and target
    X_train = train.drop(['Status'], axis = 1)
    y_train = train['Status']
    # train the model
    model = LGBMClassifier(n_estimators = 100, random_state = 42)
    model.fit(X_train, y_train)
    # save model as pickle
    with open('../../models/model.pkl', 'wb') as file:
        pickle.dump(model, file)