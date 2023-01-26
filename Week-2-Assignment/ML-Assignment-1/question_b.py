import pandas as pd
from sklearn.svm import LinearSVC
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report
from helper_functions import read_data
from sklearn.dummy import DummyClassifier
import numpy as np

matplotlib.use('TkAgg')


# (B) PART (I)
def part_one():
    # Read data from CSV.
    X1, X2, X, y, y_train = read_data()

    # Train linear SVM classifiers for C=[0.001, 1, 100]
    for C in [0.001, 1, 100]:
        # If the maximum iterations is 10,000, we can be sure it doesn't fail to converge for values like C=100
        model = LinearSVC(C=C, max_iter=10000).fit(X, y_train)

        # Report the parameter values of each trained model
        print("C: ", C)
        print("Intercept: ", model.intercept_)
        print("Slope: ", model.coef_)

# (B) PART (II)
def part_two():
    # Read data from CSV.
    X1, X2, X, y, y_train = read_data()

    # When i=0, we can use C=0.001
    # When i=1, we can use C=1
    # When i=2, we can use C=100
    get_C_from_loop_number = {0: 0.001, 1: 1, 2: 100}
    for i in range(3):
        # 1. CROSS-VALIDATION
        # We need to do cross-validation to see how effective our model is for new data. To do this, we can use the
        # "hold out" method by splitting our data into 80% training data to train our model, and 20% test data to
        # evaluate prediction performance.
        # We can do this repeatedly in a loop, and choose the iteration with the lowest mean squared error (MSE).
        lowestMSE, modelWithLowestMSE, XtrainWithLowestMSE, XtestWithLowestMSE, ytrainWithLowestMSE, ytestWithLowestMSE, ypredWithLowestMSE = cross_validate_with_hold_out_method_and_multiple_splits(
            5, X, y, 0.2, get_C_from_loop_number[i], 10000)

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

        # Complex model we trained (C=0.001):
        # [[141   3]
        #  [22  34]]

        # Trivial model using dummy classifier (C=0.001):
        # [[144   0]
        #  [56   0]]

        # Complex model we trained (C=1):
        # [[132  12]
        #  [11  45]]

        # Trivial model using dummy classifier (C=1):
        # [[144   0]
        #  [56   0]]

        # Complex model we trained (C=100):
        # [[131  15]
        #  [9  45]]

        # Trivial model using dummy classifier (C=100):
        # [[146   0]
        #  [54   0]]

        # We can also use a classification report comparing the complex model against the trivial model.
        print(classification_report(ytestWithLowestMSE, ypredWithLowestMSE))
        print(classification_report(ytestWithLowestMSE, ydummy))

        # RESULTS FOR BASELINE PREDICTOR COMPARISON USING CLASSIFICATION REPORT
        # Complex model we trained (C=0.001):
        #                precision    recall  f1-score   support
        #
        #           -1       0.87      0.98      0.92       144
        #            1       0.92      0.61      0.73        56
        #
        #     accuracy                           0.88       200
        #    macro avg       0.89      0.79      0.82       200
        # weighted avg       0.88      0.88      0.87       200

        # Trivial model using dummy classifier (C=0.001):
        #                precision    recall  f1-score   support
        #
        #           -1       0.72      1.00      0.84       144
        #            1       0.00      0.00      0.00        56
        #
        #     accuracy                           0.72       200
        #    macro avg       0.36      0.50      0.42       200
        # weighted avg       0.52      0.72      0.60       200

        # Complex model we trained (C=1):
        #               precision    recall  f1-score   support
        #
        #           -1       0.92      0.95      0.94       153
        #            1       0.83      0.74      0.79        47
        #
        #     accuracy                           0.91       200
        #    macro avg       0.88      0.85      0.86       200
        # weighted avg       0.90      0.91      0.90       200

        # Trivial model using dummy classifier (C=1):
        #               precision    recall  f1-score   support
        #
        #           -1       0.77      1.00      0.87       153
        #            1       0.00      0.00      0.00        47
        #
        #     accuracy                           0.77       200
        #    macro avg       0.38      0.50      0.43       200
        # weighted avg       0.59      0.77      0.66       200

        # Complex model we trained (C=100):
        #                precision    recall  f1-score   support
        #
        #           -1       0.94      0.90      0.92       146
        #            1       0.75      0.83      0.79        54
        #
        #     accuracy                           0.88       200
        #    macro avg       0.84      0.87      0.85       200
        # weighted avg       0.89      0.88      0.88       200

        # Trivial model using dummy classifier (C=100):
        #                precision    recall  f1-score   support
        #
        #           -1       0.73      1.00      0.84       146
        #            1       0.00      0.00      0.00        54
        #
        #     accuracy                           0.73       200
        #    macro avg       0.36      0.50      0.42       200
        # weighted avg       0.53      0.73      0.62       200



        # 3. PLOT THE PREDICTIONS

        XtestAsDataFrame = pd.DataFrame(XtestWithLowestMSE)
        X1TestData = XtestAsDataFrame[0]  # X1 as test data
        X2TestData = XtestAsDataFrame[1]  # X2 as test data

        # Plot the actual target values from the data
        plt.rcParams['figure.constrained_layout.use'] = True
        plt.suptitle("SVM Models")
        # Plot iteration 0 (C=0.001), plot the data on row 3, column 1, as the 1st subplot (311)
        # Plot iteration 1 (C=1), plot the data on row 3, column 1, as the 2nd subplot (312)
        # Plot iteration 2 (C=100), plot the data on row 3, column 1, as the 3rd subplot (313)
        plt.subplot(311 + i)
        plt.title("C = " + str(get_C_from_loop_number[i]) + " (mean squared error = " + str(lowestMSE) + ")",
                  fontsize=10)
        plt.scatter(X1[y == 1], X2[y == 1], color='green', marker="+")
        plt.scatter(X1[y == -1], X2[y == -1], color='red', marker="o")
        # Plot the predictions
        plt.scatter(X1TestData[ypredWithLowestMSE == 1], X2TestData[ypredWithLowestMSE == 1], color='blue', marker="x")
        plt.scatter(X1TestData[ypredWithLowestMSE == -1], X2TestData[ypredWithLowestMSE == -1], color='black',
                    marker="x")
        plt.ylabel("X2")
        plt.xlabel("X1")

        # Calculate and plot the decision boundary
        slopes = modelWithLowestMSE.coef_.T
        theta_zero = modelWithLowestMSE.intercept_
        theta_one = slopes[0]
        theta_two = slopes[1]
        y_val = [-1 * ((theta_zero + x * theta_one) / theta_two) for x in X1]
        plt.plot(X1, y_val, linewidth=2, color='black')
        plt.legend(["+1 real", "-1 real", "+1 predicted", "-1 predicted"], loc=(1.02, 0.1), fontsize=9)
    plt.show()


def cross_validate_with_hold_out_method_and_multiple_splits(iterations_for_splits, x, y, test_size, C,
                                                            iterations_for_svm):
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
    for i in range(iterations_for_splits):
        Xtrain, Xtest, ytrain, ytest = train_test_split(x, y, test_size=test_size)
        model = LinearSVC(C=C, max_iter=iterations_for_svm).fit(Xtrain, ytrain)
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
    part_two()
