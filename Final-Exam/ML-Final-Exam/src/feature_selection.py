import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

matplotlib.use("TkAgg")


# Delete features from CSV that are not related to rating (using intuition)
def deleteUnnecessaryFeatures(updatedListingsCSV):
    data = pd.read_csv(updatedListingsCSV)
    # id column has no dependence on the ratings
    data.drop('id', inplace=True, axis=1)
    # listing_url column has no dependence on the ratings
    data.drop('listing_url', inplace=True, axis=1)
    # scrape_id column has no dependence on the ratings
    data.drop('scrape_id', inplace=True, axis=1)
    # last_scraped column has no dependence on the ratings
    data.drop('last_scraped', inplace=True, axis=1)
    # source column has no dependence on the ratings
    data.drop('source', inplace=True, axis=1)
    # name column contains details we already have from other features such as: bedrooms,
    # neighbourhood_cleansed, property_type etc
    data.drop('name', inplace=True, axis=1)
    # description column contains details we already have from other features such as: bedrooms,
    # neighbourhood_cleansed, property_type and features from review comments.
    data.drop('description', inplace=True, axis=1)
    # neighborhood_overview column contains details we already have from other features such as
    # neighbourhood_cleansed and the features from review comments.
    data.drop('neighborhood_overview', inplace=True, axis=1)
    # picture_url column has no dependence on the ratings
    data.drop('picture_url', inplace=True, axis=1)
    # host_id column has no dependence on the ratings
    data.drop('host_id', inplace=True, axis=1)
    # host_url column has no dependence on the ratings
    data.drop('host_url', inplace=True, axis=1)
    # host_about column contains details we already have from other features such as: host_response_time,
    # host_response_rate, host_acceptance rate and features from review comments.
    data.drop('host_about', inplace=True, axis=1)
    # host_response_time column has been one-hot encoded already (appended at end of CSV), so we delete this
    data.drop('host_response_time', inplace=True, axis=1)
    # host_thumbnail_url column has no dependence on the ratings
    data.drop('host_thumbnail_url', inplace=True, axis=1)
    # host_picture_url column has no dependence on the ratings
    data.drop('host_picture_url', inplace=True, axis=1)
    # host_neighbourhood column has little dependence on the rating (if it did, it would already be covered
    # by the host's actions from features host_response_time, host_response_rate, host_acceptance rate etc.)
    data.drop('host_neighbourhood', inplace=True, axis=1)
    # host_verifications column has been one-hot encoded already (appended at end of CSV), so we delete this
    data.drop('host_verifications', inplace=True, axis=1)
    # neighbourhood column is more simply described by the feature neighbourhood_cleansed, so delete this
    data.drop('neighbourhood', inplace=True, axis=1)
    # neighbourhood_cleansed column has been one-hot encoded already (appended at end of CSV), so we delete this
    data.drop('neighbourhood_cleansed', inplace=True, axis=1)
    # neighbourhood_group_cleansed column is empty, so delete this
    data.drop('neighbourhood_group_cleansed', inplace=True, axis=1)
    # property_type column has been one-hot encoded already (appended at end of CSV), so we delete this
    data.drop('property_type', inplace=True, axis=1)
    # room_type column has been one-hot encoded already (appended at end of CSV), so we delete this
    data.drop('room_type', inplace=True, axis=1)
    # bathrooms column is empty, so delete this
    data.drop('bathrooms', inplace=True, axis=1)
    # bathrooms_text column has been one-hot encoded already (appended at end of CSV), so we delete this
    data.drop('bathrooms_text', inplace=True, axis=1)
    # amenities column has been one-hot encoded already (appended at end of CSV), so we delete this
    data.drop('amenities', inplace=True, axis=1)
    # calendar_updated column is empty, so delete this
    data.drop('calendar_updated', inplace=True, axis=1)
    # calendar_last_scraped column is not dependent on the ratings
    data.drop('calendar_last_scraped', inplace=True, axis=1)
    # license column is empty, so delete this
    data.drop('license', inplace=True, axis=1)
    data.to_csv(updatedListingsCSV, index=False, sep=',')


# Use correlation statistics to choose top 10 features for ratings
def select_features_from_correlation(X, y, dictionaryWithFeatures, location, typeOfFeatures, typeOfRatings):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    fs = SelectKBest(score_func=f_regression, k=10)
    fs.fit(X_train, y_train)
    topTenFeatures = fs.get_feature_names_out()
    plt.ylabel("Correlation feature importance", fontsize=13)
    plt.xlabel(typeOfFeatures + " features", fontsize=13)
    plt.title("Feature importance for  " + typeOfRatings, fontsize=16)
    index = 0
    for score in fs.scores_:
        if "x" + str(index) in topTenFeatures:
            plt.scatter(index, score, marker="o", s=100, c='lime')
            plt.annotate(dictionaryWithFeatures[index], (index, score), size=10)
        else:
            plt.scatter(index, score, marker="o", s=100, c='r')
        index = index + 1
    if typeOfFeatures == "Binary":
        plt.legend(loc=location, labels=["Top 10 features", "Remaining 39 features"], fontsize=12)
    else:
        plt.legend(loc=location, labels=["Top 10 features", "Remaining 44 features"], fontsize=12)
    ax = plt.gca()
    leg = ax.get_legend()
    leg.legendHandles[0].set_color('lime')
    leg.legendHandles[1].set_color('red')


# Show plots with chosen top 10 binary features and remaining 39 non-dependent features
def showBinaryFeatures(updatedListingsCSV):
    dataframe = pd.read_csv(updatedListingsCSV)

    # Potential binary features:
    multiple_hosts = dataframe.iloc[:, 0]
    host_from_ireland = dataframe.iloc[:, 2]
    host_is_superhost = dataframe.iloc[:, 5]
    host_has_profile_pic = dataframe.iloc[:, 8]
    host_identity_verified = dataframe.iloc[:, 9]
    has_availability = dataframe.iloc[:, 24]
    instant_bookable = dataframe.iloc[:, 41]
    host_respond_within_an_hour = dataframe.iloc[:, 47]
    host_respond_within_a_few_hours = dataframe.iloc[:, 48]
    host_respond_within_a_day = dataframe.iloc[:, 49]
    host_verified_by_phone = dataframe.iloc[:, 50]
    host_verified_by_email = dataframe.iloc[:, 51]
    host_verified_by_work_email = dataframe.iloc[:, 52]
    bungalow = dataframe.iloc[:, 53]
    town_house = dataframe.iloc[:, 54]
    rental_unit = dataframe.iloc[:, 55]
    home = dataframe.iloc[:, 56]
    loft = dataframe.iloc[:, 57]
    condo = dataframe.iloc[:, 58]
    cottage = dataframe.iloc[:, 59]
    guesthouse = dataframe.iloc[:, 60]
    bed_and_breakfast = dataframe.iloc[:, 61]
    boat = dataframe.iloc[:, 62]
    serviced_apartment = dataframe.iloc[:, 63]
    guest_suite = dataframe.iloc[:, 64]
    cabin = dataframe.iloc[:, 65]
    villa = dataframe.iloc[:, 66]
    castle = dataframe.iloc[:, 67]
    tiny_home = dataframe.iloc[:, 68]
    entire_home_or_apt = dataframe.iloc[:, 69]
    private_room = dataframe.iloc[:, 70]
    shared_room = dataframe.iloc[:, 71]
    shared_bath = dataframe.iloc[:, 73]
    entertainment_amenities = dataframe.iloc[:, 74]
    self_care_amenities = dataframe.iloc[:, 75]
    storage_amenities = dataframe.iloc[:, 76]
    wifi = dataframe.iloc[:, 77]
    leisure_amenities = dataframe.iloc[:, 78]
    kitchen_amenities = dataframe.iloc[:, 79]
    safety_amenities = dataframe.iloc[:, 80]
    parking_amenities = dataframe.iloc[:, 81]
    long_term_stay = dataframe.iloc[:, 82]
    single_level_home = dataframe.iloc[:, 83]
    open_24_hours = dataframe.iloc[:, 84]
    self_check_in = dataframe.iloc[:, 85]
    dublin_city = dataframe.iloc[:, 86]
    south_dublin = dataframe.iloc[:, 87]
    fingal = dataframe.iloc[:, 88]
    dun_laoghaire_rathdown = dataframe.iloc[:, 89]

    # Use abbreviated words for annotating top 10 features for feature importance plot
    binaryFeatures = dict()
    binaryFeatures[0] = "m_h"
    binaryFeatures[1] = "h_f_i"
    binaryFeatures[2] = "h_i_s"
    binaryFeatures[3] = "h_h_p_p"
    binaryFeatures[4] = "h_i_v"
    binaryFeatures[5] = "h_a"
    binaryFeatures[6] = "i_b"
    binaryFeatures[7] = "h_r_w_a_h"
    binaryFeatures[8] = "h_r_w_a_f_h"
    binaryFeatures[9] = "h_r_w_a_d"
    binaryFeatures[10] = "h_v_b_p"
    binaryFeatures[11] = "h_v_b_e"
    binaryFeatures[12] = "h_v_b_w_e"
    binaryFeatures[13] = "bun"
    binaryFeatures[14] = "t_h"
    binaryFeatures[15] = "r_u"
    binaryFeatures[16] = "home"
    binaryFeatures[17] = "loft"
    binaryFeatures[18] = "con"
    binaryFeatures[19] = "cot"
    binaryFeatures[20] = "guesthouse"
    binaryFeatures[21] = "b_a_b"
    binaryFeatures[22] = "boat"
    binaryFeatures[23] = "s_apart"
    binaryFeatures[24] = "g_s"
    binaryFeatures[25] = "cab"
    binaryFeatures[26] = "villa"
    binaryFeatures[27] = "cast"
    binaryFeatures[28] = "t_h"
    binaryFeatures[29] = "e_h_r_a"
    binaryFeatures[30] = "p_r"
    binaryFeatures[31] = "s_r"
    binaryFeatures[32] = "s_b"
    binaryFeatures[33] = "ent_am"
    binaryFeatures[34] = "s_c_a"
    binaryFeatures[35] = "stor_am"
    binaryFeatures[36] = "wifi"
    binaryFeatures[37] = "leis_am"
    binaryFeatures[38] = "kitc_am"
    binaryFeatures[39] = "saf_am"
    binaryFeatures[40] = "park_am"
    binaryFeatures[41] = "l_t_s"
    binaryFeatures[42] = "s_l_h"
    binaryFeatures[43] = "o_24"
    binaryFeatures[44] = "s_c_i"
    binaryFeatures[45] = "d_c"
    binaryFeatures[46] = "s_d"
    binaryFeatures[47] = "fing"
    binaryFeatures[48] = "d_l_r"

    # Predicting these values
    review_scores_rating = dataframe.iloc[:, 34]
    review_scores_accuracy = dataframe.iloc[:, 35]
    review_scores_cleanliness = dataframe.iloc[:, 36]
    review_scores_checkin = dataframe.iloc[:, 37]
    review_scores_communication = dataframe.iloc[:, 38]
    review_scores_location = dataframe.iloc[:, 39]
    review_scores_value = dataframe.iloc[:, 40]

    X = np.column_stack((multiple_hosts, host_from_ireland, host_is_superhost, host_has_profile_pic,
                         host_identity_verified, has_availability, instant_bookable, host_respond_within_an_hour,
                         host_respond_within_a_few_hours, host_respond_within_a_day, host_verified_by_phone,
                         host_verified_by_email, host_verified_by_work_email, bungalow, town_house, rental_unit,
                         home, loft, condo, cottage, guesthouse, bed_and_breakfast, boat, serviced_apartment,
                         guest_suite, cabin, villa, castle, tiny_home, entire_home_or_apt, private_room, shared_room,
                         shared_bath, entertainment_amenities, self_care_amenities, storage_amenities, wifi,
                         leisure_amenities, kitchen_amenities, safety_amenities, parking_amenities, long_term_stay,
                         single_level_home, open_24_hours, self_check_in, dublin_city, south_dublin, fingal,
                         dun_laoghaire_rathdown))

    y = review_scores_rating
    plt.rcParams["figure.constrained_layout.use"] = True
    plt.figure(figsize=(50, 30), dpi=80, tight_layout=True)
    plt.subplot(2, 2, 1)

    # Find correlated features for review_scores_rating
    select_features_from_correlation(X, y, binaryFeatures, "upper right", "Binary", "review_scores_rating")
    # From correlation feature selection, we see that these binary features are dependent:
    # host_from_ireland, host_is_superhost, rental_unit, home, shared_room, entertainment_amenities,
    # storage_amenities, leisure_amenities, parking_amenities, dublin_city

    # 1. Plotting binary features that do have a dependence:
    plt.subplot(2, 2, 2)

    host_from_ireland_vs_review_scores_rating = plt.scatter(host_from_ireland, review_scores_rating, marker="+",
                                                            label="host_from_IE")
    rental_unit_vs_review_scores_rating = plt.scatter(rental_unit, review_scores_rating, marker="+",
                                                      label="rental_unit")
    home_vs_review_scores_rating = plt.scatter(home, review_scores_rating, marker="+", label="home")
    shared_room_vs_review_scores_rating = plt.scatter(shared_room, review_scores_rating, marker="+",
                                                      label="shared_room")
    entertainment_amenities_vs_review_scores_rating = plt.scatter(entertainment_amenities, review_scores_rating,
                                                                  marker="+", label="entertainment_amenities")
    storage_amenities_vs_review_scores_rating = plt.scatter(storage_amenities, review_scores_rating, marker="+",
                                                            label="storage_amenities")
    leisure_amenities_vs_review_scores_rating = plt.scatter(leisure_amenities, review_scores_rating, marker="+",
                                                            label="leisure_amenities")
    parking_amenities_vs_review_scores_rating = plt.scatter(parking_amenities, review_scores_rating, marker="+",
                                                            label="parking_amenities")
    dublin_city_vs_review_scores_rating = plt.scatter(dublin_city, review_scores_rating, marker="+",
                                                      label="dublin_city")
    host_is_superhost_vs_review_scores_rating = plt.scatter(host_is_superhost, review_scores_rating, marker="+",
                                                            label="host_is_superhost")

    plt.ylabel("Review Rating", fontsize=13)
    plt.xlabel("Binary feature values", fontsize=13)
    plt.title("Top 10 dependent binary features", fontsize=16)
    plt.legend(handles=[host_from_ireland_vs_review_scores_rating,
                        host_is_superhost_vs_review_scores_rating,
                        rental_unit_vs_review_scores_rating,
                        home_vs_review_scores_rating,
                        shared_room_vs_review_scores_rating,
                        entertainment_amenities_vs_review_scores_rating,
                        storage_amenities_vs_review_scores_rating,
                        leisure_amenities_vs_review_scores_rating,
                        parking_amenities_vs_review_scores_rating,
                        dublin_city_vs_review_scores_rating], title='Legend for top 10 dependent features',
               bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=12, title_fontsize=12)

    # 2. Plotting remaining features with no/weak dependence:
    plt.subplot(2, 1, 2)

    multiple_hosts_vs_review_scores_rating = plt.scatter(multiple_hosts, review_scores_rating, marker="+",
                                                         label="multiple_hosts")

    bungalow_vs_review_scores_rating = plt.scatter(bungalow, review_scores_rating, marker="+", label="bungalow")

    loft_vs_review_scores_rating = plt.scatter(loft, review_scores_rating, marker="+", label="loft")

    cottage_vs_review_scores_rating = plt.scatter(cottage, review_scores_rating, marker="+", label="cottage")

    guesthouse_vs_review_scores_rating = plt.scatter(guesthouse, review_scores_rating, marker="+", label="guesthouse")

    guest_suite_vs_review_scores_rating = plt.scatter(guest_suite, review_scores_rating, marker="+",
                                                      label="guest_suite")

    cabin_vs_review_scores_rating = plt.scatter(cabin, review_scores_rating, marker="+", label="cabin")

    tiny_home_vs_review_scores_rating = plt.scatter(tiny_home, review_scores_rating, marker="+", label="tiny_home")

    host_identity_verified_vs_review_scores_rating = plt.scatter(host_identity_verified, review_scores_rating,
                                                                 marker="+", label="host_identity_verified")

    instant_bookable_vs_review_scores_rating = plt.scatter(instant_bookable, review_scores_rating, marker="+",
                                                           label="instant_bookable")

    host_respond_within_an_hour_vs_review_scores_rating = plt.scatter(host_respond_within_an_hour, review_scores_rating,
                                                                      marker="+", label="host_respond_within_hr")

    host_respond_within_a_few_hours_vs_review_scores_rating = plt.scatter(host_respond_within_a_few_hours,
                                                                          review_scores_rating, marker="+",
                                                                          label="host_respond_within_few_hrs")

    host_respond_within_a_day_vs_review_scores_rating = plt.scatter(host_respond_within_a_day, review_scores_rating,
                                                                    marker="+", label="host_respond_within_a_day")

    host_verified_by_email_vs_review_scores_rating = plt.scatter(host_verified_by_email, review_scores_rating,
                                                                 marker="+", label="host_verified_by_email")

    host_verified_by_work_email_vs_review_scores_rating = plt.scatter(host_verified_by_work_email, review_scores_rating,
                                                                      marker="+", label="host_verified_by_work_email")

    town_house_vs_review_scores_rating = plt.scatter(town_house, review_scores_rating, marker="+", label="town_house")

    condo_vs_review_scores_rating = plt.scatter(condo, review_scores_rating, marker="+", label="condo")

    bed_and_breakfast_vs_review_scores_rating = plt.scatter(bed_and_breakfast, review_scores_rating, marker="+",
                                                            label="bed_and_breakfast")

    boat_vs_review_scores_rating = plt.scatter(boat, review_scores_rating, marker="+", label="boat")

    serviced_apartment_vs_review_scores_rating = plt.scatter(serviced_apartment, review_scores_rating, marker="+",
                                                             label="serviced_apartment")

    villa_vs_review_scores_rating = plt.scatter(villa, review_scores_rating, marker="+", label="villa")

    castle_vs_review_scores_rating = plt.scatter(castle, review_scores_rating, marker="+", label="castle")

    entire_home_or_apt_vs_review_scores_rating = plt.scatter(entire_home_or_apt, review_scores_rating, marker="+",
                                                             label="entire_home_or_apt")

    private_room_vs_review_scores_rating = plt.scatter(private_room, review_scores_rating, marker="+",
                                                       label="private_room")

    shared_bath_vs_review_scores_rating = plt.scatter(shared_bath, review_scores_rating, marker="+",
                                                      label="shared_bath")

    self_care_amenities_vs_review_scores_rating = plt.scatter(self_care_amenities, review_scores_rating, marker="+",
                                                              label="self_care_amenities")

    wifi_vs_review_scores_rating = plt.scatter(wifi, review_scores_rating, marker="+", label="wifi")

    kitchen_amenities_vs_review_scores_rating = plt.scatter(kitchen_amenities, review_scores_rating, marker="+",
                                                            label="kitchen_amenities")

    safety_amenities_vs_review_scores_rating = plt.scatter(safety_amenities, review_scores_rating, marker="+",
                                                           label="safety_amenities")

    long_term_stay_vs_review_scores_rating = plt.scatter(long_term_stay, review_scores_rating, marker="+",
                                                         label="long_term_stay")

    single_level_home_vs_review_scores_rating = plt.scatter(single_level_home, review_scores_rating, marker="+",
                                                            label="single_level_home")

    open_24_hours_vs_review_scores_rating = plt.scatter(open_24_hours, review_scores_rating, marker="+",
                                                        label="open_24_hours")

    self_check_in_vs_review_scores_rating = plt.scatter(self_check_in, review_scores_rating, marker="+",
                                                        label="self_check_in")

    south_dublin_vs_review_scores_rating = plt.scatter(south_dublin, review_scores_rating, marker="+",
                                                       label="south_dublin")

    fingal_vs_review_scores_rating = plt.scatter(fingal, review_scores_rating, marker="+", label="fingal")

    dun_laoghaire_rathdown_vs_review_scores_rating = plt.scatter(dun_laoghaire_rathdown, review_scores_rating,
                                                                 marker="+", label="dun_laoghaire_rathdown")

    plt.ylabel("Review Rating", fontsize=13)
    plt.xlabel("Binary Features", fontsize=13)
    plt.title("Remaining non-dependent binary features", fontsize=16)
    plt.legend(handles=[multiple_hosts_vs_review_scores_rating,
                        host_is_superhost_vs_review_scores_rating,
                        bungalow_vs_review_scores_rating,
                        loft_vs_review_scores_rating,
                        cottage_vs_review_scores_rating,
                        guesthouse_vs_review_scores_rating,
                        guest_suite_vs_review_scores_rating,
                        cabin_vs_review_scores_rating,
                        tiny_home_vs_review_scores_rating,
                        host_from_ireland_vs_review_scores_rating,
                        host_identity_verified_vs_review_scores_rating,
                        instant_bookable_vs_review_scores_rating,
                        host_respond_within_an_hour_vs_review_scores_rating,
                        host_respond_within_a_few_hours_vs_review_scores_rating,
                        host_respond_within_a_day_vs_review_scores_rating,
                        host_verified_by_email_vs_review_scores_rating,
                        host_verified_by_work_email_vs_review_scores_rating,
                        town_house_vs_review_scores_rating,
                        rental_unit_vs_review_scores_rating,
                        home_vs_review_scores_rating,
                        condo_vs_review_scores_rating,
                        bed_and_breakfast_vs_review_scores_rating,
                        boat_vs_review_scores_rating,
                        serviced_apartment_vs_review_scores_rating,
                        villa_vs_review_scores_rating,
                        castle_vs_review_scores_rating,
                        entire_home_or_apt_vs_review_scores_rating,
                        private_room_vs_review_scores_rating,
                        shared_room_vs_review_scores_rating,
                        shared_bath_vs_review_scores_rating,
                        entertainment_amenities_vs_review_scores_rating,
                        self_care_amenities_vs_review_scores_rating,
                        storage_amenities_vs_review_scores_rating,
                        wifi_vs_review_scores_rating,
                        leisure_amenities_vs_review_scores_rating,
                        kitchen_amenities_vs_review_scores_rating,
                        safety_amenities_vs_review_scores_rating,
                        parking_amenities_vs_review_scores_rating,
                        long_term_stay_vs_review_scores_rating,
                        single_level_home_vs_review_scores_rating,
                        open_24_hours_vs_review_scores_rating,
                        self_check_in_vs_review_scores_rating,
                        dublin_city_vs_review_scores_rating,
                        south_dublin_vs_review_scores_rating,
                        fingal_vs_review_scores_rating,
                        dun_laoghaire_rathdown_vs_review_scores_rating],
               title='Legend for remaining non-dependent features',
               bbox_to_anchor=(1.01, 1.05), loc='upper left', fontsize=10, title_fontsize=12, ncol=2)
    plt.show()
    plt.rcParams["figure.constrained_layout.use"] = True
    plt.figure(figsize=(50, 30), dpi=80, tight_layout=True)
    plt.subplot(3, 2, 1)
    # Find correlated features for review_scores_accuracy
    y = review_scores_accuracy
    select_features_from_correlation(X, y, binaryFeatures, "upper center", "Binary", "review_scores_accuracy")
    plt.subplot(3, 2, 2)
    # Find correlated features for review_scores_cleanliness
    y = review_scores_cleanliness
    select_features_from_correlation(X, y, binaryFeatures, "upper center", "Binary", "review_scores_cleanliness")
    plt.subplot(3, 2, 3)
    # Find correlated features for review_scores_checkin
    y = review_scores_checkin
    select_features_from_correlation(X, y, binaryFeatures, "lower right", "Binary", "review_scores_checkin")
    plt.subplot(3, 2, 4)
    # Find correlated features for review_scores_communication
    y = review_scores_communication
    select_features_from_correlation(X, y, binaryFeatures, "upper center", "Binary", "review_scores_communication")
    plt.subplot(3, 2, 5)
    # Find correlated features for review_scores_location
    y = review_scores_location
    select_features_from_correlation(X, y, binaryFeatures, "upper center", "Binary", "review_scores_location")
    plt.subplot(3, 2, 6)
    # Find correlated features for review_scores_value
    y = review_scores_value
    select_features_from_correlation(X, y, binaryFeatures, "upper center", "Binary", "review_scores_value")
    plt.show()


# Show plots with chosen top 10 continuous features and remaining 44 non-dependent features
def showContinuousFeatures(updatedListingsCSV):
    # Use Min-Max scaling to normalise the data to be between 0 and 1.
    dataframe = pd.read_csv(updatedListingsCSV)
    scaler = MinMaxScaler()

    # Potential continuous features:
    host_since = scaler.fit_transform(dataframe.iloc[:, 1].values.reshape(-1, 1))
    host_response_rate = scaler.fit_transform(dataframe.iloc[:, 3].values.reshape(-1, 1))
    host_acceptance_rate = scaler.fit_transform(dataframe.iloc[:, 4].values.reshape(-1, 1))
    host_listings_count = scaler.fit_transform(dataframe.iloc[:, 6].values.reshape(-1, 1))
    host_total_listings_count = scaler.fit_transform(dataframe.iloc[:, 7].values.reshape(-1, 1))
    latitude = scaler.fit_transform(dataframe.iloc[:, 10].values.reshape(-1, 1))
    longitude = scaler.fit_transform(dataframe.iloc[:, 11].values.reshape(-1, 1))
    accommodates = scaler.fit_transform(dataframe.iloc[:, 12].values.reshape(-1, 1))
    bedrooms = scaler.fit_transform(dataframe.iloc[:, 13].values.reshape(-1, 1))
    beds = scaler.fit_transform(dataframe.iloc[:, 14].values.reshape(-1, 1))
    price = scaler.fit_transform(dataframe.iloc[:, 15].values.reshape(-1, 1))
    minimum_nights = scaler.fit_transform(dataframe.iloc[:, 16].values.reshape(-1, 1))
    maximum_nights = scaler.fit_transform(dataframe.iloc[:, 17].values.reshape(-1, 1))
    minimum_minimum_nights = scaler.fit_transform(dataframe.iloc[:, 18].values.reshape(-1, 1))
    maximum_minimum_nights = scaler.fit_transform(dataframe.iloc[:, 19].values.reshape(-1, 1))
    minimum_maximum_nights = scaler.fit_transform(dataframe.iloc[:, 20].values.reshape(-1, 1))
    maximum_maximum_nights = scaler.fit_transform(dataframe.iloc[:, 21].values.reshape(-1, 1))
    minimum_nights_avg_ntm = scaler.fit_transform(dataframe.iloc[:, 22].values.reshape(-1, 1))
    maximum_nights_avg_ntm = scaler.fit_transform(dataframe.iloc[:, 23].values.reshape(-1, 1))
    availability_30 = scaler.fit_transform(dataframe.iloc[:, 25].values.reshape(-1, 1))
    availability_60 = scaler.fit_transform(dataframe.iloc[:, 26].values.reshape(-1, 1))
    availability_90 = scaler.fit_transform(dataframe.iloc[:, 27].values.reshape(-1, 1))
    availability_365 = scaler.fit_transform(dataframe.iloc[:, 28].values.reshape(-1, 1))
    number_of_reviews = scaler.fit_transform(dataframe.iloc[:, 29].values.reshape(-1, 1))
    number_of_reviews_ltm = scaler.fit_transform(dataframe.iloc[:, 30].values.reshape(-1, 1))
    number_of_reviews_l30d = scaler.fit_transform(dataframe.iloc[:, 31].values.reshape(-1, 1))
    first_review = scaler.fit_transform(dataframe.iloc[:, 32].values.reshape(-1, 1))
    last_review = scaler.fit_transform(dataframe.iloc[:, 33].values.reshape(-1, 1))
    calculated_host_listings_count = scaler.fit_transform(dataframe.iloc[:, 42].values.reshape(-1, 1))
    calculated_host_listings_count_entire_homes = scaler.fit_transform(dataframe.iloc[:, 43].values.reshape(-1, 1))
    calculated_host_listings_count_private_rooms = scaler.fit_transform(dataframe.iloc[:, 44].values.reshape(-1, 1))
    calculated_host_listings_count_shared_rooms = scaler.fit_transform(dataframe.iloc[:, 45].values.reshape(-1, 1))
    reviews_per_month = scaler.fit_transform(dataframe.iloc[:, 46].values.reshape(-1, 1))
    number_of_bathrooms = scaler.fit_transform(dataframe.iloc[:, 72].values.reshape(-1, 1))
    city_center = dataframe.iloc[:, 90]
    clean_comfortable = dataframe.iloc[:, 91]
    definitely_recommend = dataframe.iloc[:, 92]
    definitely_stay = dataframe.iloc[:, 93]
    gave_us = dataframe.iloc[:, 94]
    great_host = dataframe.iloc[:, 95]
    great_location = dataframe.iloc[:, 96]
    great_place = dataframe.iloc[:, 97]
    great_stay = dataframe.iloc[:, 98]
    highly_recommend = dataframe.iloc[:, 99]
    location_great = dataframe.iloc[:, 100]
    minute_walk = dataframe.iloc[:, 101]
    place_great = dataframe.iloc[:, 102]
    place_stay = dataframe.iloc[:, 103]
    recommend_place = dataframe.iloc[:, 104]
    temple_bar = dataframe.iloc[:, 105]
    walking_distance = dataframe.iloc[:, 106]
    would_definitely = dataframe.iloc[:, 107]
    would_highly = dataframe.iloc[:, 108]
    would_recommend = dataframe.iloc[:, 109]

    # Use abbreviated words for annotating top 10 features for feature importance plot
    continuousFeatures = dict()
    continuousFeatures[0] = "h_s"
    continuousFeatures[1] = "h_r_r"
    continuousFeatures[2] = "h_a_r"
    continuousFeatures[3] = "h_l_c"
    continuousFeatures[4] = "h_t_l_c"
    continuousFeatures[5] = "lat"
    continuousFeatures[6] = "long"
    continuousFeatures[7] = "a"
    continuousFeatures[8] = "bedr"
    continuousFeatures[9] = "bed"
    continuousFeatures[10] = "p"
    continuousFeatures[11] = "min_n"
    continuousFeatures[12] = "max_n"
    continuousFeatures[13] = "min_min_n"
    continuousFeatures[14] = "max_min_n"
    continuousFeatures[15] = "min_max_n"
    continuousFeatures[16] = "max_max_n"
    continuousFeatures[17] = "max_n_a_n"
    continuousFeatures[18] = "min_n_a_n"
    continuousFeatures[19] = "a_30"
    continuousFeatures[20] = "a_60"
    continuousFeatures[21] = "a_90"
    continuousFeatures[22] = "a_365"
    continuousFeatures[23] = "n_o_r"
    continuousFeatures[24] = "n_o_r_lt"
    continuousFeatures[25] = "n_o_r_l30"
    continuousFeatures[26] = "f_r"
    continuousFeatures[27] = "l_r"
    continuousFeatures[28] = "c_h_l_c"
    continuousFeatures[29] = "c_h_l_c_e_h"
    continuousFeatures[30] = "c_h_l_c_p_r"
    continuousFeatures[31] = "c_h_l_c_s_r"
    continuousFeatures[32] = "r_p_m"
    continuousFeatures[33] = "n_o_b"
    continuousFeatures[34] = "city_c"
    continuousFeatures[35] = "clean_c"
    continuousFeatures[36] = "d_rec"
    continuousFeatures[37] = "d_stay"
    continuousFeatures[38] = "g_u"
    continuousFeatures[39] = "g_h"
    continuousFeatures[40] = "g_l"
    continuousFeatures[41] = "g_p"
    continuousFeatures[42] = "g_s"
    continuousFeatures[43] = "h_r"
    continuousFeatures[44] = "l_g"
    continuousFeatures[45] = "m_w"
    continuousFeatures[46] = "p_g"
    continuousFeatures[47] = "p_s"
    continuousFeatures[48] = "r_p"
    continuousFeatures[49] = "t_b"
    continuousFeatures[50] = "walk_d"
    continuousFeatures[51] = "w_def"
    continuousFeatures[52] = "w_high"
    continuousFeatures[53] = "w_rec"

    # Predicting these values
    review_scores_rating = dataframe.iloc[:, 34]
    review_scores_accuracy = dataframe.iloc[:, 35]
    review_scores_cleanliness = dataframe.iloc[:, 36]
    review_scores_checkin = dataframe.iloc[:, 37]
    review_scores_communication = dataframe.iloc[:, 38]
    review_scores_location = dataframe.iloc[:, 39]
    review_scores_value = dataframe.iloc[:, 40]

    X = np.column_stack((host_since, host_response_rate, host_acceptance_rate, host_listings_count,
                         host_total_listings_count, latitude, longitude, accommodates, bedrooms, beds, price,
                         minimum_nights, maximum_nights, minimum_minimum_nights, maximum_minimum_nights,
                         minimum_maximum_nights, maximum_maximum_nights, minimum_nights_avg_ntm,
                         maximum_nights_avg_ntm, availability_30, availability_60, availability_90, availability_365,
                         number_of_reviews, number_of_reviews_ltm, number_of_reviews_l30d, first_review, last_review,
                         calculated_host_listings_count, calculated_host_listings_count_entire_homes,
                         calculated_host_listings_count_private_rooms, calculated_host_listings_count_shared_rooms,
                         reviews_per_month, number_of_bathrooms, city_center, clean_comfortable, definitely_recommend,
                         definitely_stay, gave_us, great_host, great_location, great_place, great_stay,
                         highly_recommend,
                         location_great, minute_walk, place_great, place_stay, recommend_place, temple_bar,
                         walking_distance, would_definitely, would_highly, would_recommend))
    y = review_scores_rating

    plt.rcParams["figure.constrained_layout.use"] = True
    plt.figure(figsize=(50, 30), dpi=80, tight_layout=True)
    plt.subplot(2, 2, 1)
    # Find correlated features for review_scores_rating
    select_features_from_correlation(X, y, continuousFeatures, "upper left", "Continuous", "review_scores_rating")
    # From correlation feature selection, we see that these binary features are dependent:
    # host_since, host_response_rate, host_listings_count, host_total_listings_count, longitude, number_of_reviews,
    # last_review, calculated_host_listings_count, calculated_host_listings_count_private_rooms,
    # calculated_host_listings_count_shared_rooms

    # 1. Plot continuous features that have a dependence:
    plt.subplot(2, 2, 2)
    host_since_vs_review_scores_rating = plt.scatter(host_since, review_scores_value, marker="+", label="host_since")

    host_response_rate_vs_review_scores_rating = plt.scatter(host_response_rate, review_scores_value, marker="+",
                                                             label="host_response_rate")

    host_listings_count_vs_review_scores_rating = plt.scatter(host_listings_count, review_scores_value, marker="+",
                                                              label="host_listings_count")

    host_total_listings_count_vs_review_scores_rating = plt.scatter(host_total_listings_count, review_scores_value,
                                                                    marker="+", label="host_total_listings_count")

    longitude_vs_review_scores_rating = plt.scatter(longitude, review_scores_value, marker="+", label="longitude")

    number_of_reviews_vs_review_scores_rating = plt.scatter(number_of_reviews, review_scores_value, marker="+",
                                                            label="number_of_reviews")

    last_review_vs_review_scores_rating = plt.scatter(last_review, review_scores_value, marker="+", label="last_review")

    calculated_host_listings_count_vs_review_scores_rating = plt.scatter(calculated_host_listings_count,
                                                                         review_scores_value, marker="+",
                                                                         label="calculated_host_listings_count")

    calculated_host_listings_count_private_rooms_vs_review_scores_rating = plt.scatter(
        calculated_host_listings_count_private_rooms, review_scores_value, marker="+",
        label="calculated_host_listings_count_private_rooms")

    calculated_host_listings_count_shared_rooms_vs_review_scores_rating = plt.scatter(
        calculated_host_listings_count_shared_rooms, review_scores_value, marker="+",
        label="calculated_host_listings_count_shared_rooms")

    plt.ylabel("Review Rating", fontsize=13)
    plt.xlabel("Continuous Features", fontsize=13)
    plt.title("Dependent continuous features", fontsize=16)
    plt.legend(handles=[
        host_since_vs_review_scores_rating,
        host_response_rate_vs_review_scores_rating,
        host_listings_count_vs_review_scores_rating,
        host_total_listings_count_vs_review_scores_rating,
        longitude_vs_review_scores_rating,
        number_of_reviews_vs_review_scores_rating,
        last_review_vs_review_scores_rating,
        calculated_host_listings_count_vs_review_scores_rating,
        calculated_host_listings_count_private_rooms_vs_review_scores_rating,
        calculated_host_listings_count_shared_rooms_vs_review_scores_rating],
        title='Legend for 10 dependent features', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=12,
        title_fontsize=12, ncol=1)

    # 2 Plot remaining features with no/weak dependence:
    plt.subplot(2, 1, 2)

    city_center_vs_review_scores_rating = plt.scatter(city_center, review_scores_value, marker="+", label="city_center")

    clean_comfortable_vs_review_scores_rating = plt.scatter(clean_comfortable, review_scores_value, marker="+",
                                                            label="clean_comfortable")

    definitely_recommend_vs_review_scores_rating = plt.scatter(definitely_recommend, review_scores_value, marker="+",
                                                               label="definitely_recommend")

    definitely_stay_vs_review_scores_rating = plt.scatter(definitely_stay, review_scores_value, marker="+",
                                                          label="definitely_stay")

    gave_us_vs_review_scores_rating = plt.scatter(gave_us, review_scores_value, marker="+", label="gave_us")

    great_host_vs_review_scores_rating = plt.scatter(great_host, review_scores_value, marker="+", label="great_host")

    great_location_vs_review_scores_rating = plt.scatter(great_location, review_scores_value, marker="+",
                                                         label="great_location")

    great_place_vs_review_scores_rating = plt.scatter(great_place, review_scores_value, marker="+", label="great_place")

    great_stay_vs_review_scores_rating = plt.scatter(great_stay, review_scores_value, marker="+", label="great_stay")

    highly_recommend_vs_review_scores_rating = plt.scatter(highly_recommend, review_scores_value, marker="+",
                                                           label="highly_recommend")

    location_great_vs_review_scores_rating = plt.scatter(location_great, review_scores_value, marker="+",
                                                         label="location_great")

    minute_walk_vs_review_scores_rating = plt.scatter(minute_walk, review_scores_value, marker="+", label="minute_walk")

    place_great_vs_review_scores_rating = plt.scatter(place_great, review_scores_value, marker="+", label="place_great")

    place_stay_vs_review_scores_rating = plt.scatter(place_stay, review_scores_value, marker="+", label="place_stay")

    recommend_place_vs_review_scores_rating = plt.scatter(recommend_place, review_scores_value, marker="+",
                                                          label="recommend_place")

    temple_bar_vs_review_scores_rating = plt.scatter(temple_bar, review_scores_value, marker="+", label="temple_bar")

    walking_distance_vs_review_scores_rating = plt.scatter(walking_distance, review_scores_value, marker="+",
                                                           label="walking_distance")

    would_definitely_vs_review_scores_rating = plt.scatter(would_definitely, review_scores_value, marker="+",
                                                           label="would_definitely")

    would_highly_vs_review_scores_rating = plt.scatter(would_highly, review_scores_value, marker="+",
                                                       label="would_highly")

    would_recommend_vs_review_scores_rating = plt.scatter(would_recommend, review_scores_value, marker="+",
                                                          label="would_recommend")

    latitude_vs_review_scores_rating = plt.scatter(latitude, review_scores_value, marker="+", label="latitude")

    accommodates_vs_review_scores_rating = plt.scatter(accommodates, review_scores_value, marker="+",
                                                       label="accommodates")

    bedrooms_vs_review_scores_rating = plt.scatter(bedrooms, review_scores_value, marker="+", label="bedrooms")

    beds_vs_review_scores_rating = plt.scatter(beds, review_scores_value, marker="+", label="beds")

    availability_30_vs_review_scores_rating = plt.scatter(availability_30, review_scores_value, marker="+",
                                                          label="availability_30")

    number_of_reviews_l30d_vs_review_scores_rating = plt.scatter(number_of_reviews_l30d, review_scores_value,
                                                                 marker="+", label="number_of_reviews_l30d")

    number_of_bathrooms_vs_review_scores_rating = plt.scatter(number_of_bathrooms, review_scores_value, marker="+",
                                                              label="number_of_bathrooms")

    host_acceptance_rate_vs_review_scores_rating = plt.scatter(host_acceptance_rate, review_scores_value, marker="+",
                                                               label="host_acceptance_rate")

    price_vs_review_scores_rating = plt.scatter(price, review_scores_value, marker="+", label="price")

    minimum_nights_vs_review_scores_rating = plt.scatter(minimum_nights, review_scores_value, marker="+",
                                                         label="minimum_nights")

    maximum_nights_vs_review_scores_rating = plt.scatter(maximum_nights, review_scores_value, marker="+",
                                                         label="maximum_nights")

    minimum_minimum_nights_vs_review_scores_rating = plt.scatter(minimum_minimum_nights, review_scores_value,
                                                                 marker="+", label="minimum_minimum_nights")

    maximum_minimum_nights_vs_review_scores_rating = plt.scatter(maximum_minimum_nights, review_scores_value,
                                                                 marker="+", label="maximum_minimum_nights")

    minimum_maximum_nights_vs_review_scores_rating = plt.scatter(minimum_maximum_nights, review_scores_value,
                                                                 marker="+", label="minimum_maximum_nights")

    maximum_maximum_nights_vs_review_scores_rating = plt.scatter(maximum_maximum_nights, review_scores_value,
                                                                 marker="+", label="maximum_maximum_nights")

    minimum_nights_avg_ntm_vs_review_scores_rating = plt.scatter(minimum_nights_avg_ntm, review_scores_value,
                                                                 marker="+", label="minimum_nights_avg_ntm")

    maximum_nights_avg_ntm_vs_review_scores_rating = plt.scatter(maximum_nights_avg_ntm, review_scores_value,
                                                                 marker="+", label="maximum_nights_avg_ntm")

    availability_60_vs_review_scores_rating = plt.scatter(availability_60, review_scores_value, marker="+",
                                                          label="availability_60")

    availability_90_vs_review_scores_rating = plt.scatter(availability_90, review_scores_value, marker="+",
                                                          label="availability_90")

    availability_365_vs_review_scores_rating = plt.scatter(availability_365, review_scores_value, marker="+",
                                                           label="availability_365")

    number_of_reviews_ltm_vs_review_scores_rating = plt.scatter(number_of_reviews_ltm, review_scores_value, marker="+",
                                                                label="number_of_reviews_ltm")

    first_review_vs_review_scores_rating = plt.scatter(first_review, review_scores_value, marker="+",
                                                       label="first_review")

    calculated_host_listings_count_entire_homes_vs_review_scores_rating = plt.scatter(
        calculated_host_listings_count_entire_homes, review_scores_value, marker="+",
        label="calculated_host_listings_count_entire_homes")

    reviews_per_month_vs_review_scores_rating = plt.scatter(reviews_per_month, review_scores_value, marker="+",
                                                            label="reviews_per_month")

    plt.ylabel("Review Rating", fontsize=13)
    plt.xlabel("Continuous Features", fontsize=13)
    plt.title("Remaining non-dependent continuous features", fontsize=16)
    plt.legend(handles=[city_center_vs_review_scores_rating,
                        clean_comfortable_vs_review_scores_rating,
                        definitely_recommend_vs_review_scores_rating,
                        definitely_stay_vs_review_scores_rating,
                        gave_us_vs_review_scores_rating,
                        great_host_vs_review_scores_rating,
                        great_location_vs_review_scores_rating,
                        great_place_vs_review_scores_rating,
                        great_stay_vs_review_scores_rating,
                        highly_recommend_vs_review_scores_rating,
                        location_great_vs_review_scores_rating,
                        minute_walk_vs_review_scores_rating,
                        place_great_vs_review_scores_rating,
                        place_stay_vs_review_scores_rating,
                        recommend_place_vs_review_scores_rating,
                        temple_bar_vs_review_scores_rating,
                        walking_distance_vs_review_scores_rating,
                        would_definitely_vs_review_scores_rating,
                        would_highly_vs_review_scores_rating,
                        would_recommend_vs_review_scores_rating,
                        latitude_vs_review_scores_rating,
                        accommodates_vs_review_scores_rating,
                        bedrooms_vs_review_scores_rating,
                        beds_vs_review_scores_rating,
                        availability_30_vs_review_scores_rating,
                        number_of_reviews_l30d_vs_review_scores_rating,
                        number_of_bathrooms_vs_review_scores_rating,
                        host_since_vs_review_scores_rating,
                        host_response_rate_vs_review_scores_rating,
                        host_listings_count_vs_review_scores_rating,
                        host_acceptance_rate_vs_review_scores_rating,
                        host_total_listings_count_vs_review_scores_rating,
                        price_vs_review_scores_rating,
                        minimum_nights_vs_review_scores_rating,
                        maximum_nights_vs_review_scores_rating,
                        minimum_minimum_nights_vs_review_scores_rating,
                        maximum_minimum_nights_vs_review_scores_rating,
                        minimum_maximum_nights_vs_review_scores_rating,
                        maximum_maximum_nights_vs_review_scores_rating,
                        minimum_nights_avg_ntm_vs_review_scores_rating,
                        maximum_nights_avg_ntm_vs_review_scores_rating,
                        availability_60_vs_review_scores_rating,
                        availability_90_vs_review_scores_rating,
                        availability_365_vs_review_scores_rating,
                        number_of_reviews_ltm_vs_review_scores_rating,
                        first_review_vs_review_scores_rating,
                        last_review_vs_review_scores_rating,
                        calculated_host_listings_count_vs_review_scores_rating,
                        calculated_host_listings_count_entire_homes_vs_review_scores_rating,
                        calculated_host_listings_count_private_rooms_vs_review_scores_rating,
                        calculated_host_listings_count_shared_rooms_vs_review_scores_rating,
                        calculated_host_listings_count_shared_rooms_vs_review_scores_rating,
                        reviews_per_month_vs_review_scores_rating], title='Legend for non-dependent features',
               bbox_to_anchor=(1.01, 1.05), loc='upper left', fontsize=8.5, title_fontsize=12, ncol=2)
    plt.show()
    plt.rcParams["figure.constrained_layout.use"] = True
    plt.figure(figsize=(50, 30), dpi=80, tight_layout=True)
    plt.subplot(3, 2, 1)
    # Find correlated features for review_scores_accuracy
    y = review_scores_accuracy
    select_features_from_correlation(X, y, continuousFeatures, "upper right", "Continuous", "review_scores_accuracy")
    plt.subplot(3, 2, 2)
    # Find correlated features for review_scores_cleanliness
    y = review_scores_cleanliness
    select_features_from_correlation(X, y, continuousFeatures, "upper right", "Continuous",
                                     "review_scores_cleanliness")
    plt.subplot(3, 2, 3)
    # Find correlated features for review_scores_checkin
    y = review_scores_checkin
    select_features_from_correlation(X, y, continuousFeatures, "upper right", "Continuous", "review_scores_checkin")
    plt.subplot(3, 2, 4)
    # Find correlated features for review_scores_communication
    y = review_scores_communication
    select_features_from_correlation(X, y, continuousFeatures, "upper right", "Continuous",
                                     "review_scores_communication")
    plt.subplot(3, 2, 5)
    # Find correlated features for review_scores_location
    y = review_scores_location
    select_features_from_correlation(X, y, continuousFeatures, "upper right", "Continuous", "review_scores_location")
    plt.subplot(3, 2, 6)
    # Find correlated features for review_scores_value
    y = review_scores_value
    select_features_from_correlation(X, y, continuousFeatures, "upper right", "Continuous", "review_scores_value")
    plt.show()