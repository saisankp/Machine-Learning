import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap
from sklearn.dummy import DummyClassifier
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.patches as mpatches
from helper_functions import read_data
from sklearn.linear_model import LogisticRegression

matplotlib.use("TkAgg")


# (a) Plot the data before starting to get an idea of the data we're dealing with
def plot_dataset(X, y):
    plt.rcParams["figure.constrained_layout.use"] = True
    X_positive = X[np.where(y == 1)]
    X_negative = X[np.where(y == -1)]
    plt.scatter(X_positive[:, 0], X_positive[:, 1], c="green", marker="+", label="target value = 1")
    plt.scatter(X_negative[:, 0], X_negative[:, 1], c="red", marker="x", label="target value = -1")
    plt.ylabel("X2", fontsize=12)
    plt.xlabel("X1", fontsize=12)
    plt.title("Visualization of data with scatter plot", fontsize=17)
    plt.legend(loc="lower right", fontsize=17, markerscale=1.5)
    plt.show()
    plt.rcParams["figure.constrained_layout.use"] = False


# (a) Part (i)
def part_a_i(X, y, range_of_values_for_q, range_of_values_of_C):
    plt.rcParams["figure.constrained_layout.use"] = True
    plt.figure(dpi=120)
    for C in range_of_values_of_C:
        model = LogisticRegression(C=C, penalty="l2", max_iter=1000)
        std_error = []
        mean_error = []
        # Use 5-fold cross validation to choose q.
        for q in range_of_values_for_q:
            Xpolynomial = PolynomialFeatures(q).fit_transform(X)
            scores = cross_val_score(model, Xpolynomial, y, cv=5, scoring="f1")
            std_error.append(np.array(scores).std())
            mean_error.append(np.array(scores).mean())
        plt.errorbar(range_of_values_for_q, mean_error, yerr=std_error, label=("C = " + str(C)), linewidth=3)
    plt.title("Logistic Regression - F1 Score vs q", fontsize=17)
    plt.ylabel("F1 Score", fontsize=14)
    plt.xlabel("q", fontsize=14)
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    plt.show()
    plt.rcParams["figure.constrained_layout.use"] = False


# (a) Part (ii)
def part_a_ii(X, y, q, range_of_values_of_C):
    plt.figure(dpi=120)
    Xpolynomial = PolynomialFeatures(q).fit_transform(X)
    std_error = []
    mean_error = []
    # Use 5-fold cross validation to choose C.
    for C in range_of_values_of_C:
        model = LogisticRegression(C=C, penalty="l2", max_iter=1000)
        scores = cross_val_score(model, Xpolynomial, y, cv=5, scoring="f1")
        std_error.append(np.array(scores).std())
        mean_error.append(np.array(scores).mean())
        # Print parameters from model
        model.fit(Xpolynomial, y)
        print("C =" + str(C))
        print("θ =" + str(np.insert(model.coef_, 0, model.intercept_)))
    plt.errorbar(range_of_values_of_C, mean_error, yerr=std_error, linewidth=3)
    plt.title("Logistic Regression - F1 Score vs C", fontsize=17)
    plt.ylabel("F1 Score", fontsize=12)
    plt.xlabel("C", fontsize=12)
    plt.show()


# (b)
def part_b(X, y, q, range_of_values_of_k):
    plt.figure(dpi=120)
    Xpoly = PolynomialFeatures(q).fit_transform(X)
    std_error = []
    mean_error = []
    # Use 5-fold cross validation to choose k.
    for k in range_of_values_of_k:
        model = KNeighborsClassifier(n_neighbors=k, weights="uniform")
        scores = cross_val_score(model, Xpoly, y, cv=5, scoring="f1")
        std_error.append(np.array(scores).std())
        mean_error.append(np.array(scores).mean())
    plt.errorbar(range_of_values_of_k, mean_error, yerr=std_error, linewidth=3)
    plt.title("kNN - F1 Score vs k using q=" + str(q), fontsize=17)
    plt.ylabel("F1 Score", fontsize=12)
    plt.xlabel("k", fontsize=12)
    plt.show()


# (c)

# Split the data into testing and training with an 20/80 split.
def part_c_helper(X, y, q):
    Xpolynomial = PolynomialFeatures(q).fit_transform(X)
    Xtrain, Xtest, ytrain, ytest = train_test_split(Xpolynomial, y, test_size=0.2)
    return Xtrain, Xtest, ytrain, ytest


# Use the same data split from the part_c_helper function to plot the confusion matrices for different models.
def part_c(Xtrain, Xtest, ytrain, ytest, model, title):
    model.fit(Xtrain, ytrain)
    ConfusionMatrixDisplay.from_estimator(model, Xtest, ytest, display_labels=["y = -1", "y = 1"], cmap=plt.cm.YlOrRd)
    plt.title("Confusion matrix - " + title)
    plt.show()


# (d)
def part_d(X, y, q, C, k):
    plt.figure(dpi=120)
    Xpolynomial = PolynomialFeatures(q).fit_transform(X)
    Xtrain, Xtest, ytrain, ytest = train_test_split(Xpolynomial, y, test_size=0.2)
    # Plot ROC curve for logistic regression
    model = LogisticRegression(penalty="l2", C=C).fit(Xtrain, ytrain)
    fpr, tpr, _ = roc_curve(ytest, model.decision_function(Xtest))
    plt.plot(fpr, tpr, label="Logistic Regression (q = " + str(q) + " and C = " + str(C) + ")", color="red")
    # Plot ROC curve for kNN
    model = KNeighborsClassifier(n_neighbors=k, weights="uniform").fit(Xtrain, ytrain)
    fpr, tpr, _ = roc_curve(ytest, model.predict_proba(Xtest)[:, 1])
    plt.plot(fpr, tpr, label="kNN (k = " + str(k) + ")", color="green")
    # Plot ROC curve for dummy classifier (predicts most frequent class)
    model = DummyClassifier(strategy="most_frequent").fit(Xtrain, ytrain)
    fpr, tpr, _ = roc_curve(ytest, model.predict_proba(Xtest)[:, 1])
    plt.plot(fpr, tpr, label="Baseline Classifier (predicts most frequent class)", color="darkorange")
    # Plot ROC curve for dummy classifier (predicts randomly)
    model = DummyClassifier(strategy="uniform").fit(Xtrain, ytrain)
    fpr, tpr, _ = roc_curve(ytest, model.predict_proba(Xtest)[:, 1])
    plt.plot(fpr, tpr, label="Baseline Classifier (predicts randomly)", color="black", alpha=0.5)
    plt.title("ROC Curves for Logistic Regression, kNN, and Baseline classifiers")
    plt.ylabel("True positive rate", fontsize=12)
    plt.xlabel("False positive rate", fontsize=12)
    plt.legend(fontsize=9)
    plt.show()


# Plot the prediction surfaces for (a) and (b)
def plot_prediction_surface(X, y, q, model):
    plt.figure(dpi=120)
    Xpolynomial = PolynomialFeatures(q).fit_transform(X)
    Xtrain, Xtest, ytrain, ytest = train_test_split(Xpolynomial, y, test_size=0.2)
    Xtest_negative = Xtest[np.where(ytest == -1)]
    Xtest_positive = Xtest[np.where(ytest == 1)]
    model.fit(Xtrain, ytrain)
    # Get the minimum and maximum of X and y to create a meshgrid with colors representing the decision boundary.
    Xtest_y_minimum = Xtest[:, 2].min() - 1
    Xtest_y_maximum = Xtest[:, 2].max() + 1
    Xtest_X_minimum = Xtest[:, 1].min() - 1
    Xtest_X_maximum = Xtest[:, 1].max() + 1
    X, Y = np.meshgrid(np.arange(Xtest_X_minimum, Xtest_X_maximum, 0.01),
                       np.arange(Xtest_y_minimum, Xtest_y_maximum, 0.01))
    Xtest = PolynomialFeatures(q).fit_transform(np.c_[X.ravel(), Y.ravel()])
    predictions = model.predict(Xtest).reshape(X.shape)
    if model.get_params().__contains__("C"):
        plt.title("Logistic regression - prediction surface on test data with q=" + str(q) + " and C=" + str(
            model.get_params()["C"]))
    elif model.get_params().__contains__("n_neighbors"):
        plt.title("Prediction surface on test data (kNN) using k=" + str(model.get_params()["n_neighbors"]))
    colors_for_prediction_surface = ListedColormap(["lightcoral", "lightgreen"])
    plt.pcolormesh(X, Y, predictions, cmap=colors_for_prediction_surface)
    plt.scatter(Xtest_negative[:, 1], Xtest_negative[:, 2], c="red", marker="x", label="target value = -1")
    plt.scatter(Xtest_positive[:, 1], Xtest_positive[:, 2], c="green", marker="+", label="target value = 1")
    plt.ylabel("X2", fontsize=12)
    plt.xlabel("X1", fontsize=12)
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.insert(0, mpatches.Patch(color="lightgreen", label="predicted value = 1"))
    handles.insert(0, mpatches.Patch(color="lightcoral", label="predicted value = -1"))
    plt.legend(loc="lower right", ncol=2, fontsize=10, handles=handles)
    plt.show()


if __name__ == '__main__':
    # First data set
    X, y = read_data("week4-1.csv")

    # (a) Part (i)

    plot_dataset(X,y)
    # Choose best value for hyperparameter q with cross validation
    part_a_i(X, y, [1, 2, 3], [0.01, 1, 1000])

    # Plot predictions to see if q=2 is truly the best value for q.
    model = LogisticRegression(penalty="l2", C=1)
    plot_prediction_surface(X, y, 1, model)
    plot_prediction_surface(X, y, 2, model)

    # (a) Part (ii)

    # Choose best value for hyperparameter C with cross validation
    part_a_ii(X, y, 2, [1, 5, 10, 50, 100, 150, 250, 450, 650, 850, 1000])

    # Plot predictions to see if C=450 is truly the best value for C.
    model = LogisticRegression(penalty="l2", C=1)
    plot_prediction_surface(X, y, 2, model)
    model = LogisticRegression(penalty="l2", C=450)
    plot_prediction_surface(X, y, 2, model)

    # (b)

    part_b(X, y, 2, [1,5,15,25,35,45,55])
    model = KNeighborsClassifier(n_neighbors=25, weights="uniform")
    plot_prediction_surface(X, y, 2, model)

    # (c)

    Xtrain, Xtest, ytrain, ytest = part_c_helper(X, y, 2)
    part_c(Xtrain, Xtest, ytrain, ytest, LogisticRegression(penalty="l2", C=450), "Logistic Regression using q = 2 and C = 450")
    part_c(Xtrain, Xtest, ytrain, ytest, KNeighborsClassifier(n_neighbors=25, weights="uniform"), "kNN using k = 25")
    part_c(Xtrain, Xtest, ytrain, ytest, DummyClassifier(strategy="most_frequent"), "Baseline predicting most frequent class")
    part_c(Xtrain, Xtest, ytrain, ytest, DummyClassifier(strategy="uniform"), "Baseline predicting randomly")

    # (d)

    part_d(X, y, 2, 450, 25)

    # Second data set
    X, y = read_data("week4-2.csv")

    # (a) Part (i)

    plot_dataset(X,y)
    # Choose best value for hyperparameter q with cross validation
    part_a_i(X, y, [1, 2, 3, 4, 5], [0.001, 1, 1000])
    # Plot predictions to see if q=1 is truly the best value for q.
    model = LogisticRegression(penalty="l2", C=1)
    plot_prediction_surface(X, y, 1, model)
    plot_prediction_surface(X, y, 2, model)

    # (a) Part (ii)

    # Choose best value for hyperparameter C with cross validation
    part_a_ii(X, y, 1, [1, 5, 10, 50, 100, 150, 250, 450, 650, 850, 1000])

    # Plot predictions to see if C=1 is truly the best value for C.
    model = LogisticRegression(penalty="l2", C=1)
    plot_prediction_surface(X, y, 1, model)
    model = LogisticRegression(penalty="l2", C=450)
    plot_prediction_surface(X, y, 1, model)

    # (b)

    part_b(X, y, 1, [1, 10, 40, 70, 100, 130, 160, 190])
    model = KNeighborsClassifier(n_neighbors=70, weights="uniform")
    plot_prediction_surface(X, y, 1, model)

    # (c)

    Xtrain, Xtest, ytrain, ytest = part_c_helper(X, y, 1)
    part_c(Xtrain, Xtest, ytrain, ytest, LogisticRegression(penalty="l2", C=450), "Logistic Regression using q = 1 and C = 1")
    part_c(Xtrain, Xtest, ytrain, ytest, KNeighborsClassifier(n_neighbors=70, weights="uniform"), "kNN (k = 70)")
    part_c(Xtrain, Xtest, ytrain, ytest, DummyClassifier(strategy="most_frequent"), "Baseline Classifier (predicts most frequent class)")
    part_c(Xtrain, Xtest, ytrain, ytest, DummyClassifier(strategy="uniform"), "Baseline Classifier (predicts randomly)")

    # (d)

    part_d(X, y, q=1, C=1, k=70)
