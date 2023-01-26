import warnings
import matplotlib
from sklearn.dummy import DummyClassifier
from itertools import cycle
import numpy as np
from tensorflow.python.keras import layers, regularizers
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, LeakyReLU
import time

matplotlib.use('TkAgg')


def convolutional_network(n=None, compareWithBaseline=None, epochs=None, recordTime=None, L1_values=None,
                          showOnlyAccuracyPlot=None, useMaxPooling=None, thinnerAndDeeper=None):
    # Set all default values for variables if they are not passed in.
    if n is None:
        n = 5000
    if compareWithBaseline is None:
        compareWithBaseline = False
    if epochs is None:
        epochs = 20
    if recordTime is None:
        recordTime = False
    if L1_values is None:
        L1_values = [0.0001]
    if showOnlyAccuracyPlot is None:
        showOnlyAccuracyPlot = False
    if useMaxPooling is None:
        useMaxPooling = False
    if thinnerAndDeeper is None:
        thinnerAndDeeper = False

    # Using this command to ensure the graphs fit properly in the plot without overlapping
    plt.rcParams['figure.constrained_layout.use'] = True
    plt.rc('font', size=16)
    for L1 in L1_values:
        # Model / data parameters
        num_classes = 10
        input_shape = (32, 32, 3)

        # the data, split between train and test sets
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

        x_train = x_train[1:n];
        y_train = y_train[1:n]
        # x_test=x_test[1:500]; y_test=y_test[1:500]

        # Scale images to the [0, 1] range
        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255
        print("orig x_train shape:", x_train.shape)

        y_train_before_categorical = y_train
        y_test_before_categorical = y_test

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

        use_saved_model = False
        if use_saved_model:
            model = keras.models.load_model("cifar.model")
        else:
            model = keras.Sequential()

            if useMaxPooling:
                model.add(Conv2D(16, (3, 3), padding='same', input_shape=x_train.shape[1:], activation='relu'))
                model.add(Conv2D(16, (3, 3), padding='same', input_shape=x_train.shape[1:], activation='relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))
                model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
                model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))
            elif thinnerAndDeeper:
                model.add(Conv2D(8, (3, 3), padding='same', input_shape=x_train.shape[1:], activation='relu'))
                model.add(Conv2D(8, (3, 3), strides=(2, 2), padding='same', activation='relu'))
                model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
                model.add(Conv2D(16, (3, 3), strides=(2, 2), padding='same', activation='relu'))
                model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
                model.add(Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu'))
            elif (not useMaxPooling) and (not thinnerAndDeeper):
                model.add(Conv2D(16, (3, 3), padding='same', input_shape=x_train.shape[1:], activation='relu'))
                model.add(Conv2D(16, (3, 3), strides=(2, 2), padding='same', activation='relu'))
                model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
                model.add(Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu'))

            model.add(Dropout(0.5))
            model.add(Flatten())
            model.add(Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l1(L1)))
            model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
            model.summary()
            batch_size = 128

            if recordTime:
                time_before_training = time.time()
                history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
                print("Time taken to train network: " + str(time.time() - time_before_training))
            else:
                history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

            model.save("cifar.model")

            if showOnlyAccuracyPlot:
                plt.plot(history.history['accuracy'], label='train (L1=' + str(L1) + ')')
                plt.plot(history.history['val_accuracy'], label='val (L1=' + str(L1) + ')')
                plt.title('model accuracy (n=' + str(n) + ')')
                plt.ylabel('accuracy')
                plt.xlabel('epoch')
                plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
            else:
                plt.subplot(211)
                plt.plot(history.history['accuracy'], label='train (L1=' + str(L1) + ')')
                plt.plot(history.history['val_accuracy'], label='val (L1=' + str(L1) + ')')
                plt.title('model accuracy (n=' + str(n) + ')')
                plt.ylabel('accuracy')
                plt.xlabel('epoch')
                plt.legend(loc='upper left')
                plt.subplot(212)
                plt.plot(history.history['loss'], label='train (L1=' + str(L1) + ')')
                plt.plot(history.history['val_loss'], label='val (L1=' + str(L1) + ')')
                plt.title('model loss (n=' + str(n) + ')')
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.legend(loc='upper left')

    plt.show()

    preds = model.predict(x_train)
    y_pred = np.argmax(preds, axis=1)
    y_train1 = np.argmax(y_train, axis=1)
    # checkPerformanceWithTrainingData = Dictionary containing {classification report, confusion matrix}
    checkPerformanceWithTrainingData = {"classification-report": classification_report(y_train1, y_pred),
                                        "confusion-matrix": confusion_matrix(y_train1, y_pred)}
    print(checkPerformanceWithTrainingData["classification-report"])
    print(checkPerformanceWithTrainingData["confusion-matrix"])

    preds = model.predict(x_test)
    y_pred = np.argmax(preds, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    # checkPerformanceWithTestData = Dictionary containing {classification report, confusion matrix}
    checkPerformanceWithTestData = {"classification-report": classification_report(y_test1, y_pred),
                                    "confusion-matrix": confusion_matrix(y_test1, y_pred)}
    print(checkPerformanceWithTestData["classification-report"])
    print(checkPerformanceWithTestData["confusion-matrix"])

    if compareWithBaseline:
        # Check how this performance compares with a simple baseline that always predicts the most common label
        flattened_x_train = flatten_matrix(x_train)
        flattened_x_test = flatten_matrix(x_test)
        dummy_clf = DummyClassifier(strategy="most_frequent")
        dummy_clf.fit(flattened_x_train, y_train_before_categorical)
        DummyClassifierMeanAccuracy = dummy_clf.score(flattened_x_test, y_test_before_categorical)
        print("Dummy classifier's mean accuracy: " + str(DummyClassifierMeanAccuracy))
        return checkPerformanceWithTrainingData, checkPerformanceWithTestData, DummyClassifierMeanAccuracy
    else:
        return checkPerformanceWithTrainingData, checkPerformanceWithTestData


def flatten_matrix(input_matrix):
    output_array = []
    for i in range(input_matrix.shape[0]):
        output_array.append(input_matrix[i].flatten(order='C'))
    return np.array(output_array)


if __name__ == '__main__':
    # This assignment is based off on provided code from https://www.scss.tcd.ie/Doug.Leith/CSU44061/week8.py
    # Uncomment the part you wish to run below

    # PART (B) (I)
    # Keras says this model has 37,146 parameters.
    # The layer with the most parameters is the last layer (conv2d_3) with 9248 parameters.
    convolutional_network(n=5000, compareWithBaseline=True)

    # PART (B) (II)
    # From the plots in function convolutional_network() above, we can see over-fitting since the training and
    # validation accuracy diverges when epoch is around 15 and greater.

    # PART (B) (III)
    # convolutional_network(n=5000, recordTime=True)
    # convolutional_network(n=10000, recordTime=True)
    # convolutional_network(n=20000, recordTime=True)
    # convolutional_network(n=40000, recordTime=True)

    # PART (B) (IV)
    # convolutional_network(n=5000, L1_values=[0, 0.0001, 0.001, 0.01, 1, 50], showOnlyAccuracyPlot = True)
    # convolutional_network(n=40000, L1_values=[0.0001, 0.001, 0.01], showOnlyAccuracyPlot=True)

    # PART (C) (I) AND (II)
    # convolutional_network(n=5000, recordTime=True, useMaxPooling=True)
    # convolutional_network(n=40000, recordTime=True, useMaxPooling=True)

    # PART (D)
    # convolutional_network(n=5000, recordTime=True, thinnerAndDeeper=True)
    # convolutional_network(n=5000, recordTime=True, epochs=50, thinnerAndDeeper=True)
    # convolutional_network(n=40000, recordTime=True, epochs=50, thinnerAndDeeper=True)
