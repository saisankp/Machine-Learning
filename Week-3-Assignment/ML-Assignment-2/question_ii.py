from helper_functions import read_data
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

matplotlib.use('TkAgg')


def part_a():
    # Read in data
    X1, X2, X, y, y_train = read_data()
    Xpolynomial = PolynomialFeatures(5).fit_transform(X)
    plt.figure(dpi=120)
    TrainingDataStandardError = []
    TrainingDataMeanError = []
    TestingDataStandardError = []
    TestingDataMeanError = []
    range_of_values_for_C = [1, 10, 20, 50, 100]

    # Do 5-Fold cross validation
    kf = KFold(n_splits=5)
    for C in range_of_values_for_C:
        TrainingTemp = []
        TestingTemp = []
        model = Lasso(alpha=1 / (2 * C))
        for train, test in kf.split(Xpolynomial):
            model.fit(Xpolynomial[train], y[train])
            ypred_test = model.predict(Xpolynomial[test])
            ypred_train = model.predict(Xpolynomial[train])
            TestingTemp.append(mean_squared_error(y[test], ypred_test))
            TrainingTemp.append(mean_squared_error(y[train], ypred_train))
        TrainingDataStandardError.append(np.array(TrainingTemp).std())
        TrainingDataMeanError.append(np.array(TrainingTemp).mean())
        TestingDataStandardError.append(np.array(TestingTemp).std())
        TestingDataMeanError.append(np.array(TestingTemp).mean())

    # Plot the mean and standard deviation of prediction error vs C
    plt.title("Lasso Regression - Mean and standard deviation of prediction error vs C")
    plt.ylabel("Mean square error")
    plt.xlabel("C")
    plt.errorbar(range_of_values_for_C, TestingDataMeanError, yerr=TestingDataStandardError, linewidth=3, c="red",
                 label="Test data")
    plt.errorbar(range_of_values_for_C, TrainingDataMeanError, yerr=TrainingDataStandardError, linewidth=3, c="green",
                 label="Training data")
    plt.legend()
    plt.show()


def part_c():
    # Read in data
    X1, X2, X, y, y_train = read_data()
    Xpolynomial = PolynomialFeatures(5).fit_transform(X)
    plt.figure(dpi=120)
    TrainingDataStandardError = []
    TrainingDataMeanError = []
    TestingDataStandardError = []
    TestingDataMeanError = []
    range_of_values_for_C = [0.00001, 0.0001, 0.005, 0.01, 0.05, 0.1]

    # Do 5-Fold cross validation
    kf = KFold(n_splits=5)
    for C in range_of_values_for_C:
        TrainingTemp = []
        TestingTemp = []
        model = Ridge(alpha=1 / (2 * C))
        for train, test in kf.split(Xpolynomial):
            model.fit(Xpolynomial[train], y[train])
            ypred_test = model.predict(Xpolynomial[test])
            ypred_train = model.predict(Xpolynomial[train])
            TestingTemp.append(mean_squared_error(y[test], ypred_test))
            TrainingTemp.append(mean_squared_error(y[train], ypred_train))
        TrainingDataStandardError.append(np.array(TrainingTemp).std())
        TrainingDataMeanError.append(np.array(TrainingTemp).mean())
        TestingDataStandardError.append(np.array(TestingTemp).std())
        TestingDataMeanError.append(np.array(TestingTemp).mean())

    # Plot the mean and standard deviation of prediction error vs C
    plt.title("Ridge Regression - Mean and standard deviation of prediction error vs C")
    plt.ylabel("Mean square error")
    plt.xlabel("C")
    plt.errorbar(range_of_values_for_C, TestingDataMeanError, yerr=TestingDataStandardError, linewidth=3, c="red",
                 label="Test data")
    plt.errorbar(range_of_values_for_C, TrainingDataMeanError, yerr=TrainingDataStandardError, linewidth=3, c="green",
                 label="Training data")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # Uncomment one function call to run different parts of the answers
    # part_a()
    part_c()
