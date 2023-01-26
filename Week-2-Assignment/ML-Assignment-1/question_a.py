import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report
from sklearn.dummy import DummyClassifier
import matplotlib
import matplotlib.pyplot as plt
from helper_functions import read_data


matplotlib.use('TkAgg')


# (A) PART (I)
def part_one():
    # Read data from CSV.
    X1, X2, X, y, y_train = read_data()

    # Plot data as scatter plot
    plt.rcParams['figure.constrained_layout.use'] = True
    plt.scatter(X1[y == 1], X2[y == 1], color='green', marker="+")
    plt.scatter(X1[y == -1], X2[y == -1], color='red', marker="o")
    plt.title("Visualization of data with scatter plot", fontsize=17)
    plt.ylabel("X2", fontsize=16)
    plt.xlabel("X1", fontsize=16)
    plt.legend(["-1", "+1"], loc='lower right', fontsize=14, markerscale=1.5)
    plt.xticks(np.arange(-1, 1.01, 0.25), fontsize=14)
    plt.yticks(np.arange(-1, 1.01, 0.25), fontsize=14)
    plt.show()


# (A) PART (II)
def part_two():
    # Read data from CSV.
    X1, X2, X, y, y_train = read_data()

    # Train a logistic regression classifier on the data
    model = LogisticRegression(solver='lbfgs', penalty="none")
    model.fit(X, y_train)

    # Report the parameter values of the trained model
    print("Intercept:", model.intercept_)
    print("Slope:", model.coef_)

    # RESULTS:
    # Intercept: = Theta 0 = [-2.05262479]
    # Slope: [[Theta 1], [Theta 2]] = [[0.11265394 5.75706825]]


# (A) PART (III)
def part_three():
    X1, X2, X, y, y_train = read_data()

    # 1. CROSS-VALIDATION
    # We need to do cross-validation to see how effective our model is for new data. To do this, we can use the "hold
    # out" method by splitting our data into 80% training data to train our model, and 20% test data to evaluate
    # prediction performance.
    # We can do this repeatedly in a loop, and choose the iteration with the lowest mean squared error (MSE).
    lowestMSE, modelWithLowestMSE, XtrainWithLowestMSE, XtestWithLowestMSE, ytrainWithLowestMSE, ytestWithLowestMSE, ypredWithLowestMSE = cross_validate_with_hold_out_method_and_multiple_splits(
        5, X, y, 0.2)

    # 2. BASELINE PREDICTOR COMPARISON
    # The goal of our model is to predict something as well as possible, so we need to do compare it with a trivial
    # baseline estimator to see whether the prediction error we got on unseen test data is good or not.

    # We can make a dummy classifier to make predictions that ignore the input features.
    dummy = DummyClassifier(strategy="most_frequent").fit(XtrainWithLowestMSE, ytrainWithLowestMSE)
    ydummy = dummy.predict(XtestWithLowestMSE)

    # We can evaluate a confusion matrix for both the complex model we have trained and the trivial model
    # using the dummy classifier.
    print(confusion_matrix(ytestWithLowestMSE, ypredWithLowestMSE))
    print(confusion_matrix(ytestWithLowestMSE, ydummy))

    # RESULTS FOR BASELINE PREDICTOR COMPARISON USING CONFUSION MATRIX
    # Format of results:
    # [[TN][FP]
    #  [FN][TP]]

    # Complex model we trained (accuracy:0.9):
    # [[130  10]
    #  [10  50]]

    # Trivial model using dummy classifier (accuracy:0.765):
    # [[153   0]
    #  [47   0]]

    # We can also use a classification report comparing the complex model against the trivial model.
    print(classification_report(ytestWithLowestMSE, ypredWithLowestMSE))
    print(classification_report(ytestWithLowestMSE, ydummy))

    # RESULTS FOR BASELINE PREDICTOR COMPARISON USING CLASSIFICATION REPORT
    # Complex model we trained:
    #                   precision  recall   f1-score  support
    #
    #           -1       0.92      0.90      0.91       136
    #            1       0.79      0.83      0.81        64
    #
    #     accuracy                           0.88       200
    #    macro avg       0.85      0.86      0.86       200
    # weighted avg       0.88      0.88      0.88       200

    # Trivial model using dummy classifier:
    #                   precision  recall   f1-score  support
    #
    #           -1       0.68      1.00      0.81       136
    #            1       0.00      0.00      0.00        64
    #
    #     accuracy                           0.68       200
    #    macro avg       0.34      0.50      0.40       200
    # weighted avg       0.46      0.68      0.55       200

    # 3. PLOT THE PREDICTIONS

    XtestAsDataFrame = pd.DataFrame(XtestWithLowestMSE)
    X1TestData = XtestAsDataFrame[0]  # X1 as test data
    X2TestData = XtestAsDataFrame[1]  # X2 as test data

    # Plot the data from part (i)
    plt.scatter(X1[y == 1], X2[y == 1], color='green', marker="+")
    plt.scatter(X1[y == -1], X2[y == -1], color='red', marker="o")
    # Plot the predictions
    plt.scatter(X1TestData[ypredWithLowestMSE == 1], X2TestData[ypredWithLowestMSE == 1], color='blue', marker="x")
    plt.scatter(X1TestData[ypredWithLowestMSE == -1], X2TestData[ypredWithLowestMSE == -1], color='black', marker="x")
    plt.title("Logistic regression with decision boundary (mean squared error = " + str(lowestMSE) + ")", fontsize=17,
              wrap=True)
    plt.ylabel("X2")
    plt.xlabel("X1")

    # Calculate and plot the decision boundary
    slopes = modelWithLowestMSE.coef_.T
    theta_zero = modelWithLowestMSE.intercept_
    theta_one = slopes[0]
    theta_two = slopes[1]
    y_val = [-1 * ((theta_zero + x * theta_one) / theta_two) for x in X1]
    plt.plot(X1, y_val, linewidth=2, color='black')
    plt.legend(["+1 real", "-1 real", "+1 predicted", "-1 predicted"], loc='lower right', ncol=2, fontsize=10)
    plt.show()


def cross_validate_with_hold_out_method_and_multiple_splits(iterations, x, y, test_size):
    # Temporarily set the data associated with the model that has the lowest MSE (mean squared error) as 0.
    temporaryLowestMSE = 0
    modelWithLowestMSE = 0
    XtrainWithLowestMSE = 0
    XtestWithLowestMSE = 0
    ytrainWithLowestMSE = 0
    ytestWithLowestMSE = 0
    ypredWithLowestMSE = 0

    # Using the hold out method, we are going to split the data into (test_size*100)% test data and ((
    # 1-test_size)*100)% training data. We can then complete cross validation by looping through multiple
    # iterations of split, and choosing the model with the least mean squared error.
    for i in range(iterations):
        Xtrain, Xtest, ytrain, ytest = train_test_split(x, y, test_size=test_size)
        model = LogisticRegression().fit(Xtrain, ytrain)
        ypred = np.sign(model.predict(Xtest))

        # If we're on the first iteration, set the current mean squared error as the lowest mean squared error
        # temporarily. Or, if we're on a subsequent iteration of the loop, we can check to see if the current
        # iteration's MSE is less than the lowest MSE from previous iterations.
        if i == 0 or mean_squared_error(ytest, ypred) < temporaryLowestMSE:
            # At this point, we are on a new iteration of training our model and have a new lowest MSE.
            # We can save all our data associated with this model.
            temporaryLowestMSE = mean_squared_error(ytest, ypred)
            modelWithLowestMSE = model
            XtrainWithLowestMSE = Xtrain
            XtestWithLowestMSE = Xtest
            ytrainWithLowestMSE = ytrain
            ytestWithLowestMSE = ytest
            ypredWithLowestMSE = ypred
    # Return the data that is associated with the iteration of training our model that has the lowest MSE.
    return temporaryLowestMSE, modelWithLowestMSE, XtrainWithLowestMSE, XtestWithLowestMSE, ytrainWithLowestMSE, ytestWithLowestMSE, ypredWithLowestMSE


if __name__ == '__main__':
    # Uncomment one function call to run different parts of the answers
    # part_one()
    # part_two()
    part_three()
