from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import json

class TextProcessing:
    @classmethod
    def preprocessing_Dataset(cls):
        (cls.aTrainData, cls.bTrainData), (cls.aTestData, cls.bTestData) = mnist.load_data()
        cls.aTrainData = cls.aTrainData.reshape(cls.aTrainData.shape[0], 28, 28, 1)
        cls.aTestData = cls.aTestData.reshape(cls.aTestData.shape[0], 28, 28, 1)

        cls.aTrainData = cls.aTrainData.astype('float32')
        cls.aTestData = cls.aTestData.astype('float32')

        cls.aTrainData/=255
        cls.aTestData/=255

        no_of_classes = 10
        cls.bTrainData = np_utils.to_categorical(cls.bTrainData, no_of_classes)
        cls.bTestData = np_utils.to_categorical(cls.bTestData, no_of_classes)

    @classmethod
    def Layer_class(cls):
        model = Sequential()
        model.add(Conv2D(filters=32,
                        # Try and change it to (5, 5)
                        kernel_size=(3, 3),
                        input_shape=(28, 28, 1)))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(BatchNormalization(axis=-1))
        # rectified linear unit
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())

        # Fully Connected Layer
        model.add(Dense(512))
        # 11th-april = this takes more time. See after completion.
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.25))
        model.add(Dense(10))

        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])

        gen = ImageDataGenerator(rotation_range=8,
                                width_shift_range=0.08,
                                shear_range=0.3,
                                height_shift_range=0.08,
                                zoom_range=0.08)

        test_gen = ImageDataGenerator()

        trainer = gen.flow(cls.aTrainData, cls.bTrainData, batch_size=128)
        tester = test_gen.flow(cls.aTestData, cls.bTestData, batch_size=128)

        model.fit_generator(trainer, steps_per_epoch=60000//128,
                            epochs=5,
                            validation_data=tester,
                            validation_steps=10000//128)

        score = model.evaluate(cls.aTestData, cls.bTestData)
        print()
        print('Test accuracy: ', score[1])

        # Second Option json
        with open("saved_model.json", "w") as outputFile:
            json.dump(model.to_json(), outputFile)
        model.save_weights("Saved_weights.h5")

tp = TextProcessing()
tp.preprocessing_Dataset()
tp.Layer_class()
