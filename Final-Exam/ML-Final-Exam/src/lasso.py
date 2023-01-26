import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score

# Comment the line below if you are not using an M1 (ARM-based) machine
from sklearn.preprocessing import PolynomialFeatures

matplotlib.use('TkAgg')


# Step 1: Selecting C for lasso model

# Selecting range of C values for the lasso model
# Conclusion from this function: Best range is between 100-300
def select_c_range(X, y):
    plt.rcParams["figure.constrained_layout.use"] = True
    mean_error = []
    std_error = []
    c_range = [1, 100, 200, 300, 400, 500, 600, 700, 800]
    for c in c_range:
        model = Lasso(alpha=1 / (2 * c))
        scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error")
        mean_error.append(abs(np.array(scores).mean()))
        std_error.append(np.array(scores).std())
    plt.errorbar(c_range, mean_error, yerr=std_error, linewidth=3)
    plt.xlabel("C")
    plt.ylabel("Mean squared error")
    plt.title("C vs Mean squared error (Selecting range for k)")
    plt.show()


# 5-fold Cross validation on range of k values selected previously (100-300)
# Conclusion from this function: Best value for C is 300
def choose_c_using_CV(X, Y):
    mean_error = []
    std_error = []
    c_range = [100, 125, 150, 175, 200, 225, 250, 275, 300]
    for c in c_range:
        model = Lasso(alpha=1 / (2 * c))
        scores = cross_val_score(model, X, Y, cv=5, scoring="neg_mean_squared_error")
        mean_error.append(abs(np.array(scores).mean()))
        std_error.append(np.array(scores).std())
    plt.errorbar(c_range, mean_error, yerr=std_error, linewidth=3)
    plt.xlabel("C")
    plt.ylabel("Mean squared error")
    plt.title("C vs Mean squared error (performing 5-fold cross-validation)")
    plt.show()


# Step 2: Selecting q for polynomial features

# 5-fold Cross validation on range of q values for polynomial features
# Conclusion from this function: Best value for q is
def choose_q_using_CV(X, Y):
    mean_error = []
    std_error = []
    q_range = [1, 2, 3, 4]
    for q in q_range:
        Xpolynomial = PolynomialFeatures(q).fit_transform(X)
        model = Lasso(alpha=1 / (2 * 300))
        scores = cross_val_score(model, Xpolynomial, Y, cv=5, scoring="neg_mean_squared_error")
        mean_error.append(abs(np.array(scores).mean()))
        std_error.append(np.array(scores).std())
    plt.errorbar(q_range, mean_error, yerr=std_error, linewidth=3)
    plt.xlabel("q")
    plt.ylabel("Mean squared error")
    plt.title("q vs Mean squared error (performing 5-fold cross-validation)")
    plt.show()


# Lasso regression model chosen hyperparameters C=300 & q=1 selected via 5-fold cross-validation
def lassoRegression(x_train, y_train):
    model = Lasso(alpha=(1 / (2 * 300))).fit(x_train, y_train)
    return model