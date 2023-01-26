from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, r2_score
from kNN import *
from lasso import *
from feature_creation import *
from feature_selection import *
import numpy as np

matplotlib.use("TkAgg")

if __name__ == '__main__':
    # Step 1: Feature engineering

    # Step 1 (A): Feature creation:
    # Concatenate all review comment text for each listing ID and store it in a list to extract features.
    listOfReviews = getListOfReviewsForEachListingID("../data/reviews.csv", "../data/listings.csv")
    # Get the features and TF-IDF weighted document-term matrix from the list of reviews.
    featuresNamesFromReviewComments, TFIDF_Matrix = featureExtraction(listOfReviews)
    # Copy the current data into another file, and while doing so we simultaneously do pre-processing and add features
    preprocessing("../data/listings.csv", "../data/updated-listings.csv", featuresNamesFromReviewComments, TFIDF_Matrix)

    # Step 1 (B): Feature selection
    # Remove useless columns that could not act as features
    deleteUnnecessaryFeatures("../data/updated-listings.csv")
    # Plot dependent vs non-dependent binary features
    showBinaryFeatures("../data/updated-listings.csv")
    # Plot dependent vs non-dependent continuous features
    showContinuousFeatures("../data/updated-listings.csv")

    # Step 2. Train machine learning models and evaluate them

    dataframe = pd.read_csv("../data/updated-listings.csv")
    scaler = MinMaxScaler()

    # Chosen binary features
    host_from_ireland = dataframe.iloc[:, 2]
    host_is_superhost = dataframe.iloc[:, 5]
    rental_unit = dataframe.iloc[:, 55]
    home = dataframe.iloc[:, 56]
    shared_room = dataframe.iloc[:, 71]
    entertainment_amenities = dataframe.iloc[:, 74]
    storage_amenities = dataframe.iloc[:, 76]
    leisure_amenities = dataframe.iloc[:, 78]
    parking_amenities = dataframe.iloc[:, 81]
    dublin_city = dataframe.iloc[:, 86]

    # Chosen continuous features
    host_since = scaler.fit_transform(dataframe.iloc[:, 1].values.reshape(-1, 1))
    host_response_rate = scaler.fit_transform(dataframe.iloc[:, 3].values.reshape(-1, 1))
    host_listings_count = scaler.fit_transform(dataframe.iloc[:, 6].values.reshape(-1, 1))
    host_total_listings_count = scaler.fit_transform(dataframe.iloc[:, 7].values.reshape(-1, 1))
    longitude = scaler.fit_transform(dataframe.iloc[:, 11].values.reshape(-1, 1))
    number_of_reviews = scaler.fit_transform(dataframe.iloc[:, 29].values.reshape(-1, 1))
    last_review = scaler.fit_transform(dataframe.iloc[:, 33].values.reshape(-1, 1))
    calculated_host_listings_count = scaler.fit_transform(dataframe.iloc[:, 42].values.reshape(-1, 1))
    calculated_host_listings_count_private_rooms = scaler.fit_transform(dataframe.iloc[:, 44].values.reshape(-1, 1))
    calculated_host_listings_count_shared_rooms = scaler.fit_transform(dataframe.iloc[:, 45].values.reshape(-1, 1))

    # Predicting these values
    review_scores_rating = dataframe.iloc[:, 34]
    review_scores_accuracy = dataframe.iloc[:, 35]
    review_scores_cleanliness = dataframe.iloc[:, 36]
    review_scores_checkin = dataframe.iloc[:, 37]
    review_scores_communication = dataframe.iloc[:, 38]
    review_scores_location = dataframe.iloc[:, 39]
    review_scores_value = dataframe.iloc[:, 40]

    X = np.column_stack((host_from_ireland, host_is_superhost, rental_unit, home, shared_room, entertainment_amenities,
                         storage_amenities, leisure_amenities, parking_amenities, dublin_city, host_since,
                         host_response_rate, host_listings_count, host_total_listings_count, longitude,
                         number_of_reviews,
                         last_review, calculated_host_listings_count, calculated_host_listings_count_private_rooms,
                         calculated_host_listings_count_shared_rooms))

    np.set_printoptions(suppress=True)

    # Tune hyperparameters (values for k, gamma, C and q end up being the same for predicting every type of rating [y])
    y = review_scores_rating

    # 5-Fold cross validation tells us we should use k=50 and gamma=1
    select_k_range(X, y)
    choose_k_using_CV(X, y)
    select_kNN_gamma_range_for_CV(X, y)
    choose_kNN_gamma_using_CV(X, y)

    # 5-Fold cross validation tells us we should use C=300 and q=1
    select_c_range(X, y)
    choose_c_using_CV(X, y)
    choose_q_using_CV(X, y)

    # Step 2 (A). Predicting review_scores_rating
    print("1. Predicting review_scores_rating:")
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = kNN(x_train, y_train)
    y_pred = model.predict(x_test)
    print("kNN MSE predicting review_scores_rating: " + str(mean_squared_error(y_test, y_pred)))
    print("kNN R-Squared predicting review_scores_rating: " + str(r2_score(np.array(y_test), np.array(y_pred))))

    model = lassoRegression(x_train, y_train)
    y_pred = model.predict(x_test)
    print("Lasso Regression MSE predicting review_scores_rating: " + str(mean_squared_error(y_test, y_pred)))
    print("Lasso Regression R-Squared predicting review_scores_rating: " + str(
        r2_score(np.array(y_test), np.array(y_pred))))

    dummyModel = DummyRegressor(strategy="mean").fit(x_train, y_train)
    y_pred = dummyModel.predict(x_test)
    print("Dummy Regressor MSE predicting review_scores_rating: " + str(mean_squared_error(y_test, y_pred)))
    print("Dummy Regressor R-Squared predicting review_scores_rating: " + str(r2_score(np.array(y_test), y_pred)))

    # Step 2 (B). Predicting review_scores_accuracy
    print("2. Predicting review_scores_accuracy:")
    y = review_scores_accuracy
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = kNN(x_train, y_train)
    y_pred = model.predict(x_test)
    print("kNN MSE predicting review_scores_rating: " + str(mean_squared_error(y_test, y_pred)))
    print("kNN R-Squared predicting review_scores_rating: " + str(r2_score(np.array(y_test), np.array(y_pred))))

    model = lassoRegression(x_train, y_train)
    y_pred = model.predict(x_test)
    print("Lasso Regression MSE predicting review_scores_rating: " + str(mean_squared_error(y_test, y_pred)))
    print("Lasso Regression R-Squared predicting review_scores_rating: " + str(
        r2_score(np.array(y_test), np.array(y_pred))))

    dummyModel = DummyRegressor(strategy="mean").fit(x_train, y_train)
    y_pred = dummyModel.predict(x_test)
    print("Dummy Regressor MSE predicting review_scores_rating: " + str(mean_squared_error(y_test, y_pred)))
    print("Dummy Regressor R-Squared predicting review_scores_rating: " + str(r2_score(np.array(y_test), y_pred)))

    # Step 2 (C). Predicting review_scores_cleanliness
    print("3. Predicting review_scores_cleanliness:")
    y = review_scores_cleanliness
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = kNN(x_train, y_train)
    y_pred = model.predict(x_test)
    print("kNN MSE predicting review_scores_cleanliness: " + str(mean_squared_error(y_test, y_pred)))
    print("kNN R-Squared predicting review_scores_cleanliness: " + str(r2_score(np.array(y_test), np.array(y_pred))))

    model = lassoRegression(x_train, y_train)
    y_pred = model.predict(x_test)
    print("Lasso Regression MSE predicting review_scores_cleanliness: " + str(mean_squared_error(y_test, y_pred)))
    print("Lasso Regression R-Squared predicting review_scores_cleanliness: " + str(
        r2_score(np.array(y_test), np.array(y_pred))))

    dummyModel = DummyRegressor(strategy="mean").fit(x_train, y_train)
    y_pred = dummyModel.predict(x_test)
    print("Dummy Regressor MSE predicting review_scores_cleanliness: " + str(mean_squared_error(y_test, y_pred)))
    print("Dummy Regressor R-Squared predicting review_scores_cleanliness: " + str(r2_score(np.array(y_test), y_pred)))

    # Step 2 (D). Predicting review_scores_checkin
    print("4. Predicting review_scores_checkin:")
    y = review_scores_checkin
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = kNN(x_train, y_train)
    y_pred = model.predict(x_test)
    print("kNN MSE predicting review_scores_checkin: " + str(mean_squared_error(y_test, y_pred)))
    print("kNN R-Squared predicting review_scores_checkin: " + str(r2_score(np.array(y_test), np.array(y_pred))))

    model = lassoRegression(x_train, y_train)
    y_pred = model.predict(x_test)
    print("Lasso Regression MSE predicting review_scores_checkin: " + str(mean_squared_error(y_test, y_pred)))
    print("Lasso Regression R-Squared predicting review_scores_checkin: " + str(
        r2_score(np.array(y_test), np.array(y_pred))))

    dummyModel = DummyRegressor(strategy="mean").fit(x_train, y_train)
    y_pred = dummyModel.predict(x_test)
    print("Dummy Regressor MSE predicting review_scores_checkin: " + str(mean_squared_error(y_test, y_pred)))
    print("Dummy Regressor R-Squared predicting review_scores_checkin: " + str(r2_score(np.array(y_test), y_pred)))

    # Step 2 (E). Predicting review_scores_communication
    print("5. Predicting review_scores_communication:")
    y = review_scores_communication
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = kNN(x_train, y_train)
    y_pred = model.predict(x_test)
    print("kNN MSE predicting review_scores_communication: " + str(mean_squared_error(y_test, y_pred)))
    print("kNN R-Squared predicting review_scores_communication: " + str(r2_score(np.array(y_test), np.array(y_pred))))

    model = lassoRegression(x_train, y_train)
    y_pred = model.predict(x_test)
    print("Lasso Regression MSE predicting review_scores_communication: " + str(mean_squared_error(y_test, y_pred)))
    print("Lasso Regression R-Squared predicting review_scores_communication: " + str(
        r2_score(np.array(y_test), np.array(y_pred))))

    dummyModel = DummyRegressor(strategy="mean").fit(x_train, y_train)
    y_pred = dummyModel.predict(x_test)
    print("Dummy Regressor MSE predicting review_scores_communication: " + str(mean_squared_error(y_test, y_pred)))
    print(
        "Dummy Regressor R-Squared predicting review_scores_communication: " + str(r2_score(np.array(y_test), y_pred)))

    # Step 2 (F). Predicting review_scores_location
    print("6. Predicting review_scores_location:")
    y = review_scores_location
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = kNN(x_train, y_train)
    y_pred = model.predict(x_test)
    print("kNN MSE predicting review_scores_location: " + str(mean_squared_error(y_test, y_pred)))
    print("kNN R-Squared predicting review_scores_location: " + str(r2_score(np.array(y_test), np.array(y_pred))))

    model = lassoRegression(x_train, y_train)
    y_pred = model.predict(x_test)
    print("Lasso Regression MSE predicting review_scores_location: " + str(mean_squared_error(y_test, y_pred)))
    print("Lasso Regression R-Squared predicting review_scores_location: " + str(
        r2_score(np.array(y_test), np.array(y_pred))))

    dummyModel = DummyRegressor(strategy="mean").fit(x_train, y_train)
    y_pred = dummyModel.predict(x_test)
    print("Dummy Regressor MSE predicting review_scores_location: " + str(mean_squared_error(y_test, y_pred)))
    print("Dummy Regressor R-Squared predicting review_scores_location: " + str(r2_score(np.array(y_test), y_pred)))

    # Step 2 (G). Predicting review_scores_location
    print("7. Predicting review_scores_value:")
    y = review_scores_value
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = kNN(x_train, y_train)
    y_pred = model.predict(x_test)
    print("kNN MSE predicting review_scores_value: " + str(mean_squared_error(y_test, y_pred)))
    print("kNN R-Squared predicting review_scores_value: " + str(r2_score(np.array(y_test), np.array(y_pred))))

    model = lassoRegression(x_train, y_train)
    y_pred = model.predict(x_test)
    print("Lasso Regression MSE predicting review_scores_value: " + str(mean_squared_error(y_test, y_pred)))
    print("Lasso Regression R-Squared predicting review_scores_value: " + str(
        r2_score(np.array(y_test), np.array(y_pred))))

    dummyModel = DummyRegressor(strategy="mean").fit(x_train, y_train)
    y_pred = dummyModel.predict(x_test)
    print("Dummy Regressor MSE predicting review_scores_value: " + str(mean_squared_error(y_test, y_pred)))
    print("Dummy Regressor R-Squared predicting review_scores_value: " + str(r2_score(np.array(y_test), y_pred)))
