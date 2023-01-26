from sklearn.feature_extraction.text import TfidfVectorizer
import emoji
from datetime import datetime
import csv
import re
import nltk
import warnings

warnings.filterwarnings("ignore")
nltk.download('words')
nltk.download('stopwords')
words = set(nltk.corpus.words.words())


# Using the review comment text for the Airbnb listings,  We need to choose 25 features (i.e. words) that
# occur in the reviews for each listing and store them in a list to add them to the updated-listings file later.
def getListOfReviewsForEachListingID(reviewsCSV, listingsCSV):
    dictionary_with_id_mapped_to_reviews = dict()
    with open(reviewsCSV) as inputFile:
        reader = csv.reader(inputFile.readlines())
        header = True
        for line in reader:
            if header:
                header = False
            else:
                reviewWithEnglishOnly = " ".join(
                    x for x in nltk.wordpunct_tokenize(str(line[5])) if x.lower() in words or not x.isalpha())
                reviewsWithCleanedText = emoji.demojize(
                    " ".join((reviewWithEnglishOnly.replace("< />", "").replace("<br/>", "")).split()))
                dictionary_with_id_mapped_to_reviews[line[0]] = str(
                    dictionary_with_id_mapped_to_reviews.get(line[0]) or "") + " " + reviewsWithCleanedText
    listOfReviewsInOrder = []
    with open(listingsCSV) as inputFile:
        reader = csv.reader(inputFile.readlines())
        header = True
        for line in reader:
            if header:
                header = False
            else:
                listOfReviewsInOrder.append(dictionary_with_id_mapped_to_reviews.get(line[0]) or "")
    return listOfReviewsInOrder


# Use the list of reviews for each listing ID to get features and a TF-IDF weighted document-term matrix.
def featureExtraction(listOfReviewsInOrder):
    vectorizer = TfidfVectorizer(norm="l2", stop_words=nltk.corpus.stopwords.words("english"), ngram_range=(2, 2),
                                 max_features=20)
    X = vectorizer.fit_transform(listOfReviewsInOrder)
    return vectorizer.get_feature_names_out(), X.toarray()


# Copy listing data to a new file while simultaneously doing pre-processing and adding previous features
def preprocessing(listingsCSV, updatedListingsCSV, featuresNamesFromReviewComments, TFIDF_Matrix):
    with open(listingsCSV) as inputFile:
        reader = csv.reader(inputFile.readlines())
    with open(updatedListingsCSV, 'w') as outputFile:
        writer = csv.writer(outputFile)
        header = True
        index = 0
        for line in reader:
            if header:
                line[11] = "multiple_hosts"
                line[13] = "host_from_ireland"
                # Extra features we will make from the existing columns with one-hot encodings.
                line.append("host_respond_within_an_hour")
                line.append("host_respond_within_a_few_hours")
                line.append("host_respond_within_a_day")
                line.append("host_verified_by_phone")
                line.append("host_verified_by_email")
                line.append("host_verified_by_work_email")
                line.append("bungalow")
                line.append("town_house")
                line.append("rental_unit")
                line.append("home")
                line.append("loft")
                line.append("condo")
                line.append("cottage")
                line.append("guesthouse")
                line.append("bed_and_breakfast")
                line.append("boat")
                line.append("serviced_apartment")
                line.append("guest_suite")
                line.append("cabin")
                line.append("villa")
                line.append("castle")
                line.append("tiny_home")
                line.append("entire_home_or_apt")
                line.append("private_room")
                line.append("shared_room")
                line.append("number_of_bathrooms")
                line.append("shared_bath")
                line.append("entertainment_amenities")
                line.append("self_care_amenities")
                line.append("storage_amenities")
                line.append("wifi")
                line.append("leisure_amenities")
                line.append("kitchen_amenities")
                line.append("safety_amenities")
                line.append("parking_amenities")
                line.append("long_term_stay")
                line.append("single_level_home")
                line.append("open_24_hours")
                line.append("self_check_in")
                line.append("dublin_city")
                line.append("south_dublin")
                line.append("fingal")
                line.append("dun_laoghaire_rathdown")
                for feature in featuresNamesFromReviewComments:
                    line.append(feature)
                header = False
            else:
                # If any of the review ratings are empty then don't keep this line (as we cannot predict anything)
                if line[61] == "" or line[62] == "" or line[63] == "" or line[64] == "" or line[65] == "" or line[
                    66] == "" or line[67] == "":
                    continue
                # We have 63 new rows from the new features, so add 0 in all their columns for now.
                for x in range(63):
                    line.append(0)
                # Do preprocessing to make data cleaner
                convertStringDateToUNIX(line)
                removeDollarAndPercentageSigns(line)
                convertColumnsToNumbers(line)
                groupAmenitiesAndOneHotEncoding(line)
                addFeatureValuesFromReviews(line, TFIDF_Matrix[index])
                index = index + 1
            writer.writerow(line)
        writer.writerows(reader)


def convertStringDateToUNIX(line):
    # Convert last_scraped column into UNIX timestamps.
    if any(chr.isdigit() for chr in line[3]):
        line[3] = datetime.fromisoformat(line[3]).timestamp()

    # Convert host_since column into UNIX timestamps.
    if any(chr.isdigit() for chr in line[12]):
        line[12] = datetime.fromisoformat(line[12]).timestamp()

    # Convert calendar_last_scraped column into UNIX timestamps.
    if any(chr.isdigit() for chr in line[55]):
        line[55] = datetime.fromisoformat(line[55]).timestamp()

    # Convert first_review column into UNIX timestamps.
    if any(chr.isdigit() for chr in line[59]):
        line[59] = datetime.fromisoformat(line[59]).timestamp()

    # Convert last_review column into UNIX timestamps.
    if any(chr.isdigit() for chr in line[60]):
        line[60] = datetime.fromisoformat(line[60]).timestamp()


def removeDollarAndPercentageSigns(line):
    # Remove dollar sign and ".00" in price column.
    if "$" in line[40]:
        line[40] = float(line[40].replace("$", "").replace(",", ""))

    # Remove percentage sign or 'N/A' in host_response_rate column.
    if "%" in line[16]:
        line[16] = float(line[16].replace("%", ""))
    elif "N/A" in line[16]:
        line[16] = 0

    # Remove percentage sign or 'N/A' in host_acceptance_rate column.
    if "%" in line[17]:
        line[17] = float(line[17].replace("%", ""))
    elif "N/A" in line[17]:
        line[17] = 0


def convertColumnsToNumbers(line):
    # Change source column to 1 or 0 (from 'city scrape' or 'previous scrape' previously).
    if line[4] == "city scrape":
        line[4] = 1
    elif line[4] == "previous scrape":
        line[4] = 0

    # Change host_name column to 1 or 0 (depending on if there are 1 or 2 hosts)
    if "And" in line[11]:
        line[11] = 1
    elif "&" in line[11]:
        line[11] = 1
    else:
        line[11] = 0

    # Change host_location column to 1 or 0 (depending on if they are from Ireland or not)
    if "Ireland" in line[13]:
        line[13] = 1
    else:
        line[13] = 0

    # Use one-hot encodings with host_response_time column to make 3 features (i.e. columns):
    # 1. host_respond_within_an_hour
    # 2. host_respond_within_a_few_hours
    # 3. host_respond_within_a_day
    # Using a binary matrix (where N/A or False is 0, and 1 is True).
    if line[15] == "within an hour":
        line[75] = 1
    elif line[15] == "within a few hours":
        line[76] = 1
    elif line[15] == "within a day":
        line[77] = 1

    # Change host_is_superhost column to 1 or 0 (from 't' [true] or 'f' [false] previously).
    if line[18] == "t":
        line[18] = 1
    elif line[18] == "f":
        line[18] = 0

    # Use one-hot encodings with host_verifications column to make 3 features (i.e. columns):
    # 1. host_verified_by_phone
    # 2. host_verified_by_email
    # 3. host_verified_by_work_email
    # Using a binary matrix (False is 0, and 1 is True).
    if "'phone'" in line[24]:
        line[78] = 1
    if "'email'" in line[24]:
        line[79] = 1
    if "'work_email'" in line[24]:
        line[80] = 1

    # Change host_has_profile_pic column to 1 or 0 (from 't' [true] or 'f' [false] previously).
    if line[25] == "t":
        line[25] = 1
    elif line[25] == "f":
        line[25] = 0

    # Change host_identity_verified column to 1 or 0 (from 't' [true] or 'f' [false] previously).
    if line[26] == "t":
        line[26] = 1
    elif line[26] == "f":
        line[26] = 0

    # Use one-hot encodings with neighbourhood_cleansed column to make 4 features (i.e. columns):
    # 1. dublin_city
    # 2. south_dublin
    # 3. fingal
    # 4. dun_laoghaire_rathdown
    if "Dublin City" in line[28]:
        line[114] = 1
    elif "South Dublin" in line[28]:
        line[115] = 1
    elif "Fingal" in line[28]:
        line[116] = 1
    elif "Dn Laoghaire-Rathdown" in line[28]:
        line[117] = 1

    # Use one-hot encodings with property_type column to make 16 features (i.e. columns):
    # 1. bungalow
    # 2. town_house
    # 3. rental_unit
    # 4. home
    # 5. loft
    # 6. condo
    # 7. cottage
    # 8. guesthouse
    # 9. bed_and_breakfast
    # 10. boat
    # 11. serviced_apartment
    # 12. guest_suite
    # 13. cabin
    # 14. villa
    # 15. castle
    # 16. tiny_home
    # Using a binary matrix (False is 0, and 1 is True).
    if "bungalow" in line[32]:
        line[81] = 1
    elif "townhouse" in line[32]:
        line[82] = 1
    elif "rental unit" in line[32]:
        line[83] = 1
    elif "Tiny home" in line[32]:
        line[96] = 1
    elif "home" in line[32]:
        line[84] = 1
    elif "loft" in line[32]:
        line[85] = 1
    elif "condo" in line[32]:
        line[86] = 1
    elif "cottage" in line[32]:
        line[87] = 1
    elif "guesthouse" in line[32]:
        line[88] = 1
    elif "bed and breakfast" in line[32]:
        line[89] = 1
    elif "boat" in line[32]:
        line[90] = 1
    elif "serviced apartment" in line[32]:
        line[91] = 1
    elif "guest suite" in line[32]:
        line[92] = 1
    elif "cabin" in line[32]:
        line[93] = 1
    elif "villa" in line[32]:
        line[94] = 1
    elif "Castle" in line[32]:
        line[95] = 1

    # Use one-hot encodings with room_type column to make 3 features (i.e. columns):
    # 1. entire_home_or_apt
    # 2. private_room
    # 3. shared_room
    # Using a binary matrix (False is 0, and 1 is True).
    if line[33] == "Entire home/apt":
        line[97] = 1
    elif line[33] == "Private room":
        line[98] = 1
    elif line[33] == "Shared room":
        line[99] = 1

    # Split bathrooms_text column to make 2 features (i.e. columns):
    # 1. number_of_bathrooms
    # 2. shared_bath
    # where number_of_bathrooms is simply the float in the bathrooms_text column, and shared_bath is a one-hot encoding.

    # Ensure the string representing baths is in a format so that we can use regex.
    if "Shared half-bath" in line[36]:
        line[36] = "0.5 shared bath"
    elif "Private half-bath" in line[36]:
        line[36] = "0.5 private bath"
    elif "Half-bath" in line[36]:
        line[36] = "0.5 bath"
    elif line[36] == "":
        line[36] = "0"

    # Remove characters and get only the number, then z`store it into the number_of_bathrooms column.
    line[100] = float(re.sub(r'[a-z]', '', line[36].lower()))
    if "shared" in line[36]:
        line[101] = 1

    # Change empty values in bedrooms column and beds column to 0.
    if line[37] == "":
        line[37] = 0
    if line[38] == "":
        line[38] = 0

    # Change has_availability column to 1 or 0 (from 't' [true] or 'f' [false] previously).
    if line[50] == "t":
        line[50] = 1
    elif line[50] == "f":
        line[50] = 0

    # Change instant_bookable column to 1 or 0 (from 't' [true] or 'f' [false] previously).
    if line[69] == "t":
        line[69] = 1
    elif line[69] == "f":
        line[69] = 0


def groupAmenitiesAndOneHotEncoding(line):
    amenitiesString = str(
        line[39][1:-1].replace('"', '').replace(", ", ",").replace("\\u2013", "").replace("\\u2019", "").replace(
            "\\u00", ""))
    amenitiesList = amenitiesString.split(",")

    # Group previously 1000+ amenities into 12 groups and use one-hot encoding to make 12 new features (i.e. columns)
    for amenity in amenitiesList:
        # 1. entertainment_amenities
        if "TV".casefold() in amenity.casefold():
            line[102] = 1
        elif "game".casefold() in amenity.casefold():
            line[102] = 1
        elif "video".casefold() in amenity.casefold():
            line[102] = 1
        elif "PS".casefold() in amenity.casefold():  # Playstation (PS3/PS4 etc)
            line[102] = 1
        elif "sound".casefold() in amenity.casefold():
            line[102] = 1
        elif "HBO".casefold() in amenity.casefold():
            line[102] = 1
        elif "chromecast".casefold() in amenity.casefold():
            line[102] = 1
        elif "netflix".casefold() in amenity.casefold():
            line[102] = 1
        elif "ping".casefold() in amenity.casefold():
            line[102] = 1
        elif "toys".casefold() in amenity.casefold():
            line[102] = 1
        elif "player".casefold() in amenity.casefold():
            line[102] = 1
        elif "roku".casefold() in amenity.casefold():
            line[102] = 1
        elif "ethernet".casefold() in amenity.casefold():
            line[102] = 1
        elif "cable".casefold() in amenity.casefold():
            line[102] = 1
        elif "piano".casefold() in amenity.casefold():
            line[102] = 1

        # 2. self_care_amenities
        elif "conditioner".casefold() in amenity.casefold():
            line[103] = 1
        elif "shampoo".casefold() in amenity.casefold():
            line[103] = 1
        elif "soap".casefold() in amenity.casefold():
            line[103] = 1
        elif "hair".casefold() in amenity.casefold():
            line[103] = 1
        elif "shower".casefold() in amenity.casefold():
            line[103] = 1
        elif "essentials".casefold() in amenity.casefold():
            line[103] = 1
        elif "bidet".casefold() in amenity.casefold():
            line[103] = 1
        elif "iron".casefold() in amenity.casefold():
            line[103] = 1
        elif "washer".casefold() in amenity.casefold():
            line[103] = 1
        elif "dryer".casefold() in amenity.casefold():
            line[103] = 1
        elif "bath".casefold() in amenity.casefold():
            line[103] = 1
        elif "Laundromat".casefold() in amenity.casefold():
            line[103] = 1
        elif "shades".casefold() in amenity.casefold():
            line[103] = 1
        elif "blanket".casefold() in amenity.casefold():
            line[103] = 1
        elif "fan".casefold() in amenity.casefold():
            line[103] = 1
        elif "hot".casefold() in amenity.casefold():
            line[103] = 1
        elif "linen".casefold() in amenity.casefold():
            line[103] = 1
        elif "comfort".casefold() in amenity.casefold():
            line[103] = 1

        # 3. storage_amenities
        elif "storage".casefold() in amenity.casefold():
            line[104] = 1
        elif "wardrobe".casefold() in amenity.casefold():
            line[104] = 1
        elif "dresser".casefold() in amenity.casefold():
            line[104] = 1
        elif "hanger".casefold() in amenity.casefold():
            line[104] = 1
        elif "table".casefold() in amenity.casefold():
            line[104] = 1
        elif "closet".casefold() in amenity.casefold():
            line[104] = 1
        elif "luggage".casefold() in amenity.casefold():
            line[104] = 1

        # 4. wifi
        elif "wifi".casefold() in amenity.casefold():
            line[105] = 1

        # 5. leisure_amenities
        elif "gym".casefold() in amenity.casefold():
            line[106] = 1
        elif "pool".casefold() in amenity.casefold():
            line[106] = 1
        elif "lake".casefold() in amenity.casefold():
            line[106] = 1
        elif "sauna".casefold() in amenity.casefold():
            line[106] = 1
        elif "tub".casefold() in amenity.casefold():  # Bathtub or hot tub
            line[106] = 1
        elif "kayak".casefold() in amenity.casefold():
            line[106] = 1
        elif "balcony".casefold() in amenity.casefold():
            line[106] = 1
        elif "AC".casefold() in amenity.casefold():
            line[106] = 1
        elif "air".casefold() in amenity.casefold():
            line[106] = 1
        elif "heat".casefold() in amenity.casefold():
            line[106] = 1
        elif "fire pit".casefold() in amenity.casefold():
            line[106] = 1
        elif "waterfront".casefold() in amenity.casefold():
            line[106] = 1
        elif "bikes".casefold() in amenity.casefold():
            line[106] = 1
        elif "boat".casefold() in amenity.casefold():
            line[106] = 1
        elif "ski".casefold() in amenity.casefold():
            line[106] = 1
        elif "babysitter".casefold() in amenity.casefold():
            line[106] = 1
        elif "staff".casefold() in amenity.casefold():
            line[106] = 1
        elif "elevator".casefold() in amenity.casefold():
            line[106] = 1
        elif "outdoor".casefold() in amenity.casefold():
            line[106] = 1
        elif "pets".casefold() in amenity.casefold():
            line[106] = 1
        elif "glass top".casefold() in amenity.casefold():
            line[106] = 1

        # 6. kitchen_amenities
        elif "fri".casefold() in amenity.casefold():  # Fridge or refrigerator
            line[107] = 1
        elif "oven".casefold() in amenity.casefold():
            line[107] = 1
        elif "stove".casefold() in amenity.casefold():
            line[107] = 1
        elif "toaster".casefold() in amenity.casefold():
            line[107] = 1
        elif "dish".casefold() in amenity.casefold():
            line[107] = 1
        elif "nespresso".casefold() in amenity.casefold():
            line[107] = 1
        elif "maker".casefold() in amenity.casefold():  # Coffee/Bread/Rice maker
            line[107] = 1
        elif "kettle".casefold() in amenity.casefold():
            line[107] = 1
        elif "glasses".casefold() in amenity.casefold():
            line[107] = 1
        elif "baking".casefold() in amenity.casefold():
            line[107] = 1
        elif "cooking".casefold() in amenity.casefold():
            line[107] = 1
        elif "coffee".casefold() in amenity.casefold():
            line[107] = 1
        elif "grill".casefold() in amenity.casefold():
            line[107] = 1
        elif "freezer".casefold() in amenity.casefold():
            line[107] = 1
        elif "barbecue".casefold() in amenity.casefold():
            line[107] = 1
        elif "microwave".casefold() in amenity.casefold():
            line[107] = 1
        elif "dining".casefold() in amenity.casefold():
            line[107] = 1
        elif "breakfast".casefold() in amenity.casefold():
            line[107] = 1
        elif "dinner".casefold() in amenity.casefold():
            line[107] = 1
        elif "kitchen".casefold() in amenity.casefold():
            line[107] = 1

        # 7. safety_amenities
        elif "alarm".casefold() in amenity.casefold():
            line[108] = 1
        elif "lock".casefold() in amenity.casefold():
            line[108] = 1
        elif "mosquito".casefold() in amenity.casefold():
            line[108] = 1
        elif "monitor".casefold() in amenity.casefold():
            line[108] = 1
        elif "guard".casefold() in amenity.casefold():
            line[108] = 1
        elif "gate".casefold() in amenity.casefold():
            line[108] = 1
        elif "first aid".casefold() in amenity.casefold():
            line[108] = 1
        elif "cover".casefold() in amenity.casefold():
            line[108] = 1
        elif "security".casefold() in amenity.casefold():
            line[108] = 1
        elif "extinguisher".casefold() in amenity.casefold():
            line[108] = 1
        elif "safe".casefold() in amenity.casefold():
            line[108] = 1
        elif "clean".casefold() in amenity.casefold():
            line[108] = 1
        elif "crib".casefold() in amenity.casefold():
            line[108] = 1
        elif "keypad".casefold() in amenity.casefold():
            line[108] = 1
        elif "Private entrance".casefold() in amenity.casefold():
            line[108] = 1

        # 8. parking_amenities
        elif "park".casefold() in amenity.casefold():
            line[109] = 1
        elif "carport".casefold() in amenity.casefold():
            line[109] = 1
        elif "garage".casefold() in amenity.casefold():
            line[109] = 1
        elif "EV charger".casefold() in amenity.casefold():
            line[109] = 1

        # 9. long_term_stay
        elif "long".casefold() in amenity.casefold():
            line[110] = 1

        # 10. single_level_home
        elif "single level" in amenity.casefold():
            line[111] = 1

        # 11. open_24_hours
        elif "open 24 hours" in amenity.casefold():
            line[112] = 1

        # 12. self_check_in
        elif "self check-in" in amenity.casefold():
            line[113] = 1


# Add the new features using common words in review comments
def addFeatureValuesFromReviews(line, values):
    for i in range(20):
        line[118] = values[0]
        line[119] = values[1]
        line[120] = values[2]
        line[121] = values[3]
        line[122] = values[4]
        line[123] = values[5]
        line[124] = values[6]
        line[125] = values[7]
        line[126] = values[8]
        line[127] = values[9]
        line[128] = values[10]
        line[129] = values[11]
        line[130] = values[12]
        line[131] = values[13]
        line[132] = values[14]
        line[133] = values[15]
        line[134] = values[16]
        line[135] = values[17]
        line[136] = values[18]
        line[137] = values[19]
