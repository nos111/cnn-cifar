from tensorflow.keras.datasets import cifar10
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models
from tensorflow.keras import layers 
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping

def compile_model(model):
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def load_data():
    (X_train, labels_train), (X_test, labels_test) = cifar10.load_data()
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    print(X_train.shape, X_test.shape)
    unique, counts = np.unique(labels_train, return_counts=True)
    print(dict(zip(unique, counts)))
    #normalise data set
    X_train = X_train / 255.
    X_test = X_test / 255.
    X_test = X_test / 300.
    #numbers to categories
    y_train = to_categorical(labels_train, 10)
    y_test = to_categorical(labels_test, 10)
    return (X_train, y_train), (X_test, y_test) 

def initialize_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(3, 3)))
    
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(3, 3)))

    model.add(layers.Flatten())
    model.add(layers.Flatten())
    model.add(layers.Dense(120, activation='relu'))
    model.add(layers.Dense(60, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))
    model.add(layers.Dense(10, activation='softmax'))
    
    return model

def train_model():
    model = initialize_model()
    model = compile_model(model)
    (X_train, y_train), (X_test, y_test)  = load_data()
    es = EarlyStopping(patience=30, verbose=1)

    history = model.fit(X_train, y_train, 
                        validation_split=0.3,
                        callbacks=[es], 
                        epochs=500, 
                        batch_size=32)
    
    with open("model.pkl", "wb") as file:
        pickle.dump(model, file)

train_model()
