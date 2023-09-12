import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import phik
import seaborn as sns

# create a function to make phik matrix
def phik_matriz(df : pd.DataFrame) -> list:
    df = df.drop(['ID'], axis = 1)
    matrix = df.phik_matrix()
    val = matrix.query('Status > 0.1').index.tolist()
    return val