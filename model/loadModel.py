import numpy as np
import keras.models
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam

from scipy.misc import imread, imresize, imshow
import tensorflow as tf

def init():
    no_of_classes = 10
    img_rows, img_cols = 28, 28
    input_shape = (img_rows, img_cols, 1)

    new_model = Sequential()
    new_model.add(Conv2D(filters=32,
                        kernel_size=(3, 3),                        input_shape=input_shape))
    new_model.add(BatchNormalization(axis=-1))
    new_model.add(Activation('relu'))
    new_model.add(Conv2D(32, (3, 3)))
    new_model.add(BatchNormalization(axis=-1))
    new_model.add(Activation('relu'))
    new_model.add(MaxPooling2D(pool_size=(2, 2)))

    new_model.add(Conv2D(64, (3, 3)))
    new_model.add(BatchNormalization(axis=-1))
    # rectified linear unit
    new_model.add(Activation('relu'))
    new_model.add(Conv2D(64, (3, 3)))
    new_model.add(BatchNormalization(axis=-1))
    new_model.add(Activation('relu'))
    new_model.add(MaxPooling2D(pool_size=(2, 2)))

    new_model.add(Flatten())

    # Fully Connected layer
    # Layers to start making prediction
    new_model.add(Dense(512))
    new_model.add(BatchNormalization())
    new_model.add(Activation('relu'))
    new_model.add(Dropout(0.25))
    new_model.add(Dense(10))

    new_model.add(Activation('softmax'))

    #load weights into new model
    new_model.load_weights("Saved_weights.h5")
    print("Loaded Model from disk")

    # Compile and evaluate loaded model
    new_model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])
    #loss,accuracy = model.evaluate(X_test,y_test)
    #print('loss:', loss)
    #print('accuracy:', accuracy)

    graph = tf.get_default_graph()

    return new_model, graph
