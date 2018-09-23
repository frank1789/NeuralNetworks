#!usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.optimizers import SGD
from keras.layers.convolutional import ZeroPadding2D
from keras.layers import MaxPooling2D, Convolution2D, Activation
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from keras import backend as kbe
import numpy as np

# suppress warning and error message tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

kbe.set_image_dim_ordering('th')


class FaceRecognition(object):
    m_labels = None
    m_pred = None
    m_predictions = None
    m_train_generator = None
    m_valid_generator = None
    m_test_generator = None
    m_num_train_samples = 0
    m_num_classes = 0
    m_num_validate_samples = 0

    def __init__(self, epochs, batch_size, image_width=224, image_height=244):
        self.m_epochs = epochs
        self.m_batch_size = batch_size
        self.m_image_width = image_width
        self.m_image_height = image_height
        # instance neural network model sequential
        self.m_model = Sequential()

    @staticmethod
    # Get count of number of files in this folder and all subfolders
    def get_num_files(path):
        if not os.path.exists(path):
            return 0
        return sum([len(files) for r, d, files in os.walk(path)])

    @staticmethod
    # Get count of number of sub-folders directly below the folder in path
    def get_num_subfolders(path):
        if not os.path.exists(path):
            return 0
        return sum([len(d) for r, d, files in os.walk(path)])

    @staticmethod
    def create_img_generator(rotation_range=30, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,
                             zoom_range=0.2, horizontal_flip=True):
        """Define image generators that will variations of image with the image rotated slightly, shifted up, down,
        left, or right, sheared, zoomed in, or flipped horizontally on the vertical axis (ie. person looking to the
        left ends up looking to the right)
        :return: ImageDataGenerator (object)
        """
        return ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            shear_range=shear_range,
            zoom_range=zoom_range,
            horizontal_flip=horizontal_flip
        )

    def set_train_generator(self, train_folder):
        """Connect the image generator to a folder contains the source images the image generator alters."""
        #   Training image generator
        self.m_num_train_samples = self.get_num_files(train_folder)
        self.m_num_classes = self.get_num_subfolders(train_folder)
        train_generator = self.create_img_generator()
        self.m_train_generator = train_generator.flow_from_directory(
            directory=train_folder,
            target_size=(self.m_image_width, self.m_image_height),
            batch_size=self.m_batch_size,
            seed=42  # set seed for reproducibility
        )

    def set_valid_generator(self, validate_folder):
        """Validation image generator."""
        self.m_num_validate_samples = self.get_num_files(validate_folder)
        valid_generator = self.create_img_generator()
        self.m_valid_generator = valid_generator.flow_from_directory(
            directory=validate_folder,
            target_size=(self.m_image_width, self.m_image_height),
            batch_size=self.m_batch_size,
            class_mode="categorical",
            seed=42  # set seed for reproducibility
        )

    def set_test_generator(self, test_folder):
        test_generator = self.create_img_generator()
        self.m_test_generator = test_generator.flow_from_directory(
            directory=test_folder,
            target_size=(self.m_image_width, self.m_image_height),
            color_mode="rgb",
            batch_size=1,
            class_mode=None,
            shuffle=False,
            seed=42
        )

    def tain_and_fit_model(self):
        """Train the model"""
        # Compile Neural Network
        self.face_recognition_model()
        # Fit
        STEP_SIZE_TRAIN = self.m_num_train_samples // self.m_batch_size
        STEP_SIZE_VALID = self.m_num_validate_samples // self.m_batch_size
        self.m_model.fit_generator(self.m_train_generator, steps_per_epoch=STEP_SIZE_TRAIN,
                                   validation_data=self.m_valid_generator, validation_steps=STEP_SIZE_VALID,
                                   epochs=self.m_epochs, class_weight="auto")
        # Evaluate the model
        self.m_model.evaluate_generator(generator=self.m_valid_generator)

    def predict_output(self):
        """"""
        self.m_test_generator.reset()
        self.m_pred = self.m_model.predict_generator(self.m_test_generator, verbose=0)
        # need to reset the test_generator before whenever, call the predict_generator.This is important,
        # if you forget to reset the test_generator you will get outputs in a weird order.

    def predict_class_indices(self):
        predicted_class_indices = np.argmax(self.m_pred, axis=1)
        labels = self.m_num_classes
        self.m_labels = dict((v, k) for k, v in labels.items())
        self.m_predictions = [labels[k] for k in predicted_class_indices]

    def save_json_weigth(self, path="./model/", namefile="facerec"):
        # print the scheme model
        plot_model(self.m_model, to_file='model.png')

        # save as JSON
        model_json = self.m_model.to_json()
        with open(path + namefile + ".json", "w") as json_file:
            json_file.write(model_json)
        # save weights
        self.m_model.save_weights(path + namefile + '.h5')

    def save_model(self):
        self.m_model.save("./model/facerecognition.model")
        # print the scheme model
        plot_model(self.m_model, to_file='model.png')

    def face_recognition_model(self):
        """Model face recognition based on model VGG16 Oxford University."""
        self.m_model.add(ZeroPadding2D((1, 1), input_shape=(3, self.m_image_width, self.m_image_height)))
        self.m_model.add(Convolution2D(64, (3, 3), activation='relu', padding='same'))
        self.m_model.add(ZeroPadding2D((1, 1)))
        self.m_model.add(Convolution2D(64, (3, 3), activation='relu'))
        self.m_model.add(MaxPooling2D((2, 2), data_format="channels_first", strides=(2, 2)))

        self.m_model.add(ZeroPadding2D((1, 1)))
        self.m_model.add(Convolution2D(128, (3, 3), activation='relu', padding="same"))
        self.m_model.add(ZeroPadding2D((1, 1)))
        self.m_model.add(Convolution2D(128, (3, 3), activation='relu', padding="same"))
        self.m_model.add(MaxPooling2D((2, 2), data_format="channels_first", strides=(2, 2)))

        self.m_model.add(ZeroPadding2D((1, 1)))
        self.m_model.add(Convolution2D(256, (3, 3), activation='relu'))
        self.m_model.add(ZeroPadding2D((1, 1)))
        self.m_model.add(Convolution2D(256, (3, 3), activation='relu'))
        self.m_model.add(ZeroPadding2D((1, 1)))
        self.m_model.add(Convolution2D(256, (3, 3), activation='relu'))
        self.m_model.add(MaxPooling2D((2, 2), data_format="channels_first", strides=(2, 2)))

        self.m_model.add(ZeroPadding2D((1, 1)))
        self.m_model.add(Convolution2D(512, (3, 3), activation='relu'))
        self.m_model.add(ZeroPadding2D((1, 1)))
        self.m_model.add(Convolution2D(512, (3, 3), activation='relu'))
        self.m_model.add(ZeroPadding2D((1, 1)))
        self.m_model.add(Convolution2D(512, (3, 3), activation='relu'))
        self.m_model.add(MaxPooling2D((2, 2), data_format="channels_first", strides=(2, 2)))

        self.m_model.add(ZeroPadding2D((1, 1)))
        self.m_model.add(Convolution2D(512, (3, 3), activation='relu'))
        self.m_model.add(ZeroPadding2D((1, 1)))
        self.m_model.add(Convolution2D(512, (3, 3), activation='relu'))
        self.m_model.add(ZeroPadding2D((1, 1)))
        self.m_model.add(Convolution2D(512, (3, 3), activation='relu'))
        self.m_model.add(MaxPooling2D((2, 2), data_format="channels_first", strides=(2, 2)))

        self.m_model.add(Convolution2D(4096, (7, 7), activation='relu'))
        self.m_model.add(Dropout(0.5))
        self.m_model.add(Convolution2D(4096, (1, 1), activation='relu'))
        self.m_model.add(Dropout(0.5))
        self.m_model.add(Convolution2D(2622, (1, 1)))
        self.m_model.add(Flatten())
        self.m_model.add(Activation('softmax'))
        self.m_model.add(Dense(self.m_num_classes, activation='softmax'))

        # print model structure diagram
        print(self.m_model.summary())

        # compile the  model
        self.m_model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                             loss='categorical_crossentropy', metrics=['accuracy'])


if __name__ == '__main__':
    test = FaceRecognition(epochs=10, batch_size=32)
    test.create_img_generator()
    test.set_train_generator(
        train_folder=r'/Users/francesco/Downloads/the-simpsons-characters-dataset/simpsons_dataset')
    test.set_valid_generator(
        validate_folder=r'/Users/francesco/Downloads/the-simpsons-characters-dataset/simpsons_dataset')
    test.set_test_generator(
        test_folder=r'/Users/francesco/Downloads/the-simpsons-characters-dataset/kaggle_simpson_testset')
    test.tain_and_fit_model()
    quit(0)
