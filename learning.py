import sunau
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np 
import struct
import os
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def svm(X_train, X_test, y_train, y_test):
    clf = make_pipeline(StandardScaler(), LinearSVC(max_iter=100000))
    print(clf)
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))


def NN(X_train, X_test, y_train, y_test):


    
    inputs = keras.layers.Input(shape=X_train[0, :].shape, name="input")

    x = keras.layers.Dense(256, activation="relu")(inputs)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dense(32, activation="relu")(x)

    outputs = keras.layers.Dense(10, activation="softmax", name="output")(x)

    model = keras.models.Model(inputs=inputs, outputs=outputs)


    model.summary()

    # Compile the model using Adam's default learning rate
    model.compile(optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    history = model.fit(X_train, y_train, batch_size=128, epochs=50, validation_data=(X_test, y_test))

    #print(model.weights)


    loss, acc = model.evaluate(X_test, y_test)  # returns loss and metrics
    print("loss: %.2f" % loss)
    print("acc: %.2f" % acc)



if __name__ == "__main__":
    df = pd.read_pickle('data.pkl')

    classes = ['blues', 'rock', 'metal', 'jazz', 'pop', 'disco', 'reggae', 'hiphop', 'country', 'classical']
    y, X = [], []

    for key in df:
        cat, _ = key.split('.')
        y.append(classes.index(cat))
        X.append(np.hstack(df[key]))


    y, X = np.array(y), np.array(X)
    print(y.shape, X.shape)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    std_clr = StandardScaler()
    X_train_ = std_clr.fit_transform(X_train)
    X_test_ = std_clr.transform(X_test)
    NN(X_train_, X_test_, y_train, y_test)
   
