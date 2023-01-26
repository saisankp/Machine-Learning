import matplotlib.pyplot as plt
import matplotlib
from sklearn.dummy import DummyRegressor
from helper_functions import read_data
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from matplotlib import cm

matplotlib.use('TkAgg')


def part_a():
    # Read in data
    X1, X2, X, y, y_train = read_data()

    # Plot 3D Scatter plot of data
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X1, X2, y)
    ax.set_title("3D scatter plot of training data", fontsize=17)
    ax.set_xlabel("X1", fontsize=13)
    ax.set_ylabel("X2", fontsize=13)
    ax.set_zlabel("y", fontsize=13)
    ax.set_xticks(np.arange(-1, 1.01, 0.5))
    ax.set_yticks(np.arange(-1, 1.01, 0.5))
    ax.set_zticks(np.arange(-1, 2.01, 0.5))
    plt.show()


def part_b():
    # Stop scientific notation
    np.set_printoptions(suppress=True)
    # Read in data
    X1, X2, X, y, y_train = read_data()
    Xpoly = PolynomialFeatures(5)
    XpolyFittedAndTransformed = Xpoly.fit_transform(X)
    compareWithABaseline = DummyRegressor(strategy="mean").fit(XpolyFittedAndTransformed, y)
    print("MSE using dummy regressor =", mean_squared_error(y, compareWithABaseline.predict(XpolyFittedAndTransformed)))

    # Train model for different values of C
    for C in [1, 10, 1000]:
        model = Lasso(alpha=(1 / (2 * C))).fit(XpolyFittedAndTransformed, y)
        theta = np.append(model.intercept_, model.coef_)
        print("C = ", C)
        print("MSE = ", mean_squared_error(y, model.predict(XpolyFittedAndTransformed)))
        print(Xpoly.get_feature_names_out(["X1", "X2"]))
        print("θ = ", theta)

        # RESULTS:
        # MSE using dummy regressor = 0.7038952086311983
        # C = 1
        # MSE = 0.7038952086311983
        # ['1' 'X1' 'X2' 'X1^2' 'X1 X2' 'X2^2' 'X1^3' 'X1^2 X2' 'X1 X2^2' 'X2^3'
        #  'X1^4' 'X1^3 X2' 'X1^2 X2^2' 'X1 X2^3' 'X2^4' 'X1^5' 'X1^4 X2'
        #  'X1^3 X2^2' 'X1^2 X2^3' 'X1 X2^4' 'X2^5']
        # θ = [0.6957975 0.        0.        0.        0.        0.        0.
        #      0.        0.        0.        0.        0.        0.        0.
        #      0.        0.        0.        0.        0.        0.        0.
        #      0.]
        # C = 10
        # MSE = 0.0765066721534085
        # ['1' 'X1' 'X2' 'X1^2' 'X1 X2' 'X2^2' 'X1^3' 'X1^2 X2' 'X1 X2^2' 'X2^3'
        #  'X1^4' 'X1^3 X2' 'X1^2 X2^2' 'X1 X2^3' 'X2^4' 'X1^5' 'X1^4 X2'
        #  'X1^3 X2^2' 'X1^2 X2^3' 'X1 X2^4' 'X2^5']
        # θ = [0.1777648  0.         0.         0.82572033 1.4739457  0.
        #      0.         0.         0.         0.         0.         0.
        #      0.         0.         0.         0.         0.         0.
        #      0.         0.         0.         0.]
        # C = 1000
        # MSE = 0.03844445582518942
        # ['1' 'X1' 'X2' 'X1^2' 'X1 X2' 'X2^2' 'X1^3' 'X1^2 X2' 'X1 X2^2' 'X2^3'
        #  'X1^4' 'X1^3 X2' 'X1^2 X2^2' 'X1 X2^3' 'X2^4' 'X1^5' 'X1^4 X2'
        #  'X1^3 X2^2' 'X1^2 X2^3' 'X1 X2^4' 'X2^5']
        # θ = [-0.03083267  0.          0.0312615   0.94948294  2.039377 - 0.
        #      0.06142604  0.01912658  0. - 0.04515518  0.05385162  0.
        #      - 0.          0.          0.04774425 - 0.01735093  0.          0.04684794
        #      - 0. - 0.04626425 - 0.          0.00137878]


def part_c():
    # Create grid of feature values
    Xtest = []
    grid = np.linspace(-2, 2)
    for i in grid:
        for j in grid:
            Xtest.append([i, j])

    Xtest = PolynomialFeatures(5).fit_transform(np.array(Xtest))
    X1FromXTest = Xtest[:, 1]
    X2FromXTest = Xtest[:, 2]

    # Read in data
    X1, X2, X, y, y_train = read_data()
    Xpolynomial = PolynomialFeatures(5).fit_transform(X)

    # Train model for different values of C
    for C in [1, 10, 1000]:
        model = Lasso(alpha=1 / (2 * C)).fit(Xpolynomial, y)
        # Plot the prediction surface
        fig = plt.figure(dpi=120)
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(X1.to_numpy(), X2.to_numpy(), y, color="black", label="Training data")
        plot_surface = ax.plot_trisurf(X1FromXTest, X2FromXTest, model.predict(Xtest), cmap=cm.coolwarm, alpha=0.7)
        cbar = fig.colorbar(plot_surface, shrink=0.5, aspect=5, location="left")
        cbar.ax.get_yaxis().labelpad = 10
        cbar.ax.set_ylabel('Predictions')
        ax.legend(bbox_to_anchor=(-0.4, 0.9), loc='upper left')
        ax.set_xlabel("X1", fontsize=15)
        ax.set_ylabel("X2", fontsize=15)
        ax.set_zlabel("y", fontsize=15)
        ax.set_title("Lasso predictions for C = " + str(C), fontsize=17)
        plt.show()


def part_e():
    # Create grid of feature values
    Xtest = []
    grid = np.linspace(-2, 2)
    for i in grid:
        for j in grid:
            Xtest.append([i, j])

    Xtest = PolynomialFeatures(5).fit_transform(np.array(Xtest))
    X1FromXTest = Xtest[:, 1]
    X2FromXTest = Xtest[:, 2]

    # Stop scientific notation
    np.set_printoptions(suppress=True)
    # Read in data
    X1, X2, X, y, y_train = read_data()
    Xpoly = PolynomialFeatures(5)
    XpolyFittedAndTransformed = Xpoly.fit_transform(X)
    baseline = DummyRegressor(strategy="mean").fit(XpolyFittedAndTransformed, y)
    print("MSE using dummy regressor =", mean_squared_error(y, baseline.predict(XpolyFittedAndTransformed)))

    # Train model for different values of C
    for C in [0.00001, 0.1, 1]:
        model = Ridge(alpha=1 / (2 * C)).fit(XpolyFittedAndTransformed, y)
        theta = np.append(model.intercept_, model.coef_)
        print("C =", np.format_float_positional(C, trim='-'))
        print("MSE = ", mean_squared_error(y, model.predict(XpolyFittedAndTransformed)))
        print(Xpoly.get_feature_names_out(["X1", "X2"]))
        print("θ =", theta)

        fig = plt.figure(dpi=120)
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(X1.to_numpy(), X2.to_numpy(), y, color="black", label="Training data")
        plot_surface = ax.plot_trisurf(X1FromXTest, X2FromXTest, model.predict(Xtest), cmap=cm.coolwarm, alpha=0.7)
        cbar = fig.colorbar(plot_surface, shrink=0.5, aspect=5, location="left")
        cbar.ax.get_yaxis().labelpad = 10
        cbar.ax.set_ylabel('Predictions')
        ax.legend(bbox_to_anchor=(-0.4, 0.9), loc='upper left')
        ax.set_xlabel("X1", fontsize=15)
        ax.set_ylabel("X2", fontsize=15)
        ax.set_zlabel("y", fontsize=15)
        ax.set_title("Ridge predictions for C = " + np.format_float_positional(C, trim='-'), fontsize=17)
        plt.show()

        # RESULTS:
        # MSE using dummy regressor = 0.7038952086311983
        # C = 0.00001
        # MSE =  0.7020154338896712
        # ['1' 'X1' 'X2' 'X1^2' 'X1 X2' 'X2^2' 'X1^3' 'X1^2 X2' 'X1 X2^2' 'X2^3'
        #  'X1^4' 'X1^3 X2' 'X1^2 X2^2' 'X1 X2^3' 'X2^4' 'X1^5' 'X1^4 X2'
        #  'X1^3 X2^2' 'X1^2 X2^3' 'X1 X2^4' 'X2^5']
        # θ = [0.6952409  0.         0.00009691 0.0012057  0.00070615 0.00008466
        #  0.00013399 0.00007315 0.00047145 0.00004701 0.00074751 0.00060832
        #  0.0000612  0.00028762 0.00008437 0.00012088 0.00004644 0.00030498
        #  0.00004009 0.00031386 0.00005416 0.0005341 ]
        # C = 0.1
        # MSE =  0.05311416128707483
        # ['1' 'X1' 'X2' 'X1^2' 'X1 X2' 'X2^2' 'X1^3' 'X1^2 X2' 'X1 X2^2' 'X2^3'
        #  'X1^4' 'X1^3 X2' 'X1^2 X2^2' 'X1 X2^3' 'X2^4' 'X1^5' 'X1^4 X2'
        #  'X1^3 X2^2' 'X1^2 X2^3' 'X1 X2^4' 'X2^5']
        # θ = [ 0.15284132  0.          0.02987326  0.70247459  1.03625724  0.00491339
        #   0.0075098   0.02923256  0.11534977 -0.03239274  0.23304689  0.71758198
        #   0.00770816  0.23423759  0.03212909 -0.02886263  0.01191896  0.04885767
        #  -0.03645269 -0.01338245 -0.02948576  0.07594459]
        # C = 1
        # MSE =  0.03911499387712242
        # ['1' 'X1' 'X2' 'X1^2' 'X1 X2' 'X2^2' 'X1^3' 'X1^2 X2' 'X1 X2^2' 'X2^3'
        #  'X1^4' 'X1^3 X2' 'X1^2 X2^2' 'X1 X2^3' 'X2^4' 'X1^5' 'X1^4 X2'
        #  'X1^3 X2^2' 'X1^2 X2^3' 'X1 X2^4' 'X2^5']
        # θ = [ 0.02589047  0.          0.01762356  0.86485395  1.56452688 -0.03732236
        #   0.0931881   0.05105396  0.12451327 -0.01288176  0.20182663  0.46681609
        #   0.01383756  0.16011192  0.09835117 -0.10891807  0.01167576  0.06435231
        #  -0.07057372 -0.31858558 -0.01198597 -0.01550373]


if __name__ == '__main__':
    # Uncomment one function call to run different parts of the answers
    # part_a()
    # part_b()
    # part_c()
    part_e()
