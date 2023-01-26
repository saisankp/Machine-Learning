import numpy as np
import pandas as pd


def read_data():
    # Read data from CSV.
    df = pd.read_csv("week2.csv")
    X1 = df.iloc[:, 0]
    X2 = df.iloc[:, 1]
    X = np.column_stack((X1, X2))
    y = df.iloc[:, 2]
    ytrain = np.sign(y)
    return X1, X2, X, y, ytrain
