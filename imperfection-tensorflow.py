import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
import datetime
import os
import tensorflow as tf

def load_data():
    a = np.loadtxt('features.txt')
    b = np.loadtxt('labels.txt')
    X_scaled = preprocessing.scale(a)
    train_X, val_X, train_y, val_y = train_test_split(X_scaled, b, test_size=0.3, random_state=42)
    y_train = to_categorical(train_y)
    y_val = to_categorical(val_y)
    return train_X,val_X,y_train,y_val

def create_model(nrows,ncols):
    if (nrows%2==1):
        n_nrows = nrows-1
        return tf.keras.Sequential([layers.Dense(nrows, activation='relu', input_shape=(n_cols,)),
                                layers.Dense(n_nrows/2,activation='relu'),
                                layers.Dense(40,activation='relu'),
                                layers.Dense(20,activation='relu'),
                                layers.Dense(10,activation='relu'),
                                layers.Dense(2, activation='softmax')])
    else:
        return tf.keras.Sequential([layers.Dense(nrows, activation='relu', input_shape=(n_cols,)),
                                layers.Dense(nrows/2,activation='relu'),
                                layers.Dense(40,activation='relu'),
                                layers.Dense(20,activation='relu'),
                                layers.Dense(10,activation='relu'),
                                layers.Dense(2, activation='softmax')])

train_X,val_X,y_train,y_val = load_data()
n_cols = train_X.shape[1]
n_rows = train_X.shape[0]
print(n_cols,n_rows)
model = create_model(n_rows,n_cols)
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.01),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=[tf.keras.metrics.CategoricalAccuracy()])

model.fit(train_X, y_train, epochs=100, batch_size=100,validation_data=(test_X,y_val)
          ,verbose=1)
