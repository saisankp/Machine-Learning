import matplotlib
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from gaussian_kernel import *

# Comment the line below if you are not using an M1 (ARM-based) machine
matplotlib.use('TkAgg')


# Step 1: Selecting k for kNN regressor

# Selecting range of k values for the kNN model
# Conclusion from this function: Best range is between 1 and 100
def select_k_range(X, y):
    plt.rcParams["figure.constrained_layout.use"] = True
    mean_error = []
    std_error = []
    k_range = [1, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250]
    for k in k_range:
        model = KNeighborsRegressor(n_neighbors=k)
        scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error")
        mean_error.append(abs(np.array(scores).mean()))
        std_error.append(np.array(scores).std())
    plt.errorbar(k_range, mean_error, yerr=std_error, linewidth=3)
    plt.xlabel("k")
    plt.ylabel("Mean squared error")
    plt.title("k vs Mean squared error (Selecting range of k)")
    plt.show()


# 5-fold Cross validation on range of k values selected previously (1 to 100)
# Conclusion from this function: Best value for k is 50
def choose_k_using_CV(X, Y):
    mean_error = []
    std_error = []
    k_range = [1, 25, 50, 75, 100]
    for k in k_range:
        model = KNeighborsRegressor(n_neighbors=k)
        scores = cross_val_score(model, X, Y, cv=5, scoring="neg_mean_squared_error")
        mean_error.append(abs(np.array(scores).mean()))
        std_error.append(np.array(scores).std())
    plt.errorbar(k_range, mean_error, yerr=std_error, linewidth=3)
    plt.xlabel("k");
    plt.ylabel("Mean squared error")
    plt.title("k vs Mean squared error (performing 5-fold cross-validation)")
    plt.show()


# Step 2: Selecting gamma (for weights)

# Selecting range of gamma values for the kNN model
# Conclusion from this function: The best range to use CV for gamma is less than 10.
def select_kNN_gamma_range_for_CV(X, Y):
    mean_error = []
    std_error = []
    model = KNeighborsRegressor(n_neighbors=50, weights=gaussian_kernel10)
    scores = cross_val_score(model, X, Y, cv=5, scoring="neg_mean_squared_error")
    mean_error.append(abs(np.array(scores).mean()))
    std_error.append(np.array(scores).std())
    model = KNeighborsRegressor(n_neighbors=50, weights=gaussian_kernel30)
    scores = cross_val_score(model, X, Y, cv=5, scoring="neg_mean_squared_error")
    mean_error.append(abs(np.array(scores).mean()))
    std_error.append(np.array(scores).std())
    model = KNeighborsRegressor(n_neighbors=50, weights=gaussian_kernel100)
    scores = cross_val_score(model, X, Y, cv=5, scoring="neg_mean_squared_error")
    mean_error.append(abs(np.array(scores).mean()))
    std_error.append(np.array(scores).std())
    model = KNeighborsRegressor(n_neighbors=50, weights=gaussian_kernel150)
    scores = cross_val_score(model, X, Y, cv=5, scoring="neg_mean_squared_error")
    mean_error.append(abs(np.array(scores).mean()))
    std_error.append(np.array(scores).std())
    plt.errorbar([10, 30, 100, 150], mean_error, yerr=std_error, linewidth=3)
    plt.xlabel("Gamma")
    plt.ylabel("Mean squared error")
    plt.title("Gamma vs Mean squared error (Selecting range of gamma)")
    plt.show()


# 5-fold Cross validation on range of gamma values selected previously (less than 10)
# Conclusion from this function: The best value for gamma is 1.
def choose_kNN_gamma_using_CV(X, Y):
    mean_error = []
    std_error = []
    model = KNeighborsRegressor(n_neighbors=50, weights=gaussian_kernel1)
    scores = cross_val_score(model, X, Y, cv=5, scoring="neg_mean_squared_error")
    mean_error.append(abs(np.array(scores).mean()))
    std_error.append(np.array(scores).std())
    model = KNeighborsRegressor(n_neighbors=50, weights=gaussian_kernel5)
    scores = cross_val_score(model, X, Y, cv=5, scoring="neg_mean_squared_error")
    mean_error.append(abs(np.array(scores).mean()))
    std_error.append(np.array(scores).std())
    model = KNeighborsRegressor(n_neighbors=50, weights=gaussian_kernel10)
    scores = cross_val_score(model, X, Y, cv=5, scoring="neg_mean_squared_error")
    mean_error.append(abs(np.array(scores).mean()))
    std_error.append(np.array(scores).std())
    plt.errorbar([1, 5, 10], mean_error, yerr=std_error, linewidth=3)
    plt.xlabel("Gamma")
    plt.ylabel("Mean squared error")
    plt.title("Gamma vs Mean squared error (Performing 5-fold cross-validation)")
    plt.show()


# kNN model with chosen hyperparameters k=50 & gamma=1 selected via 5-fold cross-validation
def kNN(x_train, y_train):
    model_knn = KNeighborsRegressor(n_neighbors=50, weights=gaussian_kernel1).fit(x_train, y_train)
    return model_knn