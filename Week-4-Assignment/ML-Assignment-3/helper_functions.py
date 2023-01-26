import numpy as np
import pandas as pd


def read_data(Filename):
    # Read data from CSV.
    df = pd.read_csv(Filename)
    X1 = df.iloc[:, 0]
    X2 = df.iloc[:, 1]
    X = np.column_stack((X1, X2))
    y = df.iloc[:, 2]
    return X, y
