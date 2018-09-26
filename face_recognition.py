#!usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from keras.models import Model, model_from_json, load_model
from keras.layers import Dropout, Flatten, Dense, Input
from keras.optimizers import SGD
from keras.layers.convolutional import ZeroPadding2D
from keras.layers import MaxPooling2D, Convolution2D, Activation
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from keras import backend as kbe
import numpy as np
import h5py

# suppress warning and error message tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# specify input shape
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
    m_model = None

    def __init__(self, epochs, batch_size, image_width=224, image_height=224):
        self.m_epochs = epochs
        self.m_batch_size = batch_size
        self.m_image_width = image_width
        self.m_image_height = image_height

    @staticmethod
    # Get count of number of files in this folder and all sub-folders
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

    def train_and_fit_model(self):
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
        labels = self.m_train_generator.class_indices
        self.m_labels = dict((v, k) for k, v in labels.items())
        self.m_predictions = [labels[k] for k in predicted_class_indices]

    def input_model(self, model_file):
        if os.path.exists(model_file):
            self.m_model = load_model(model_file)

    def input_json_weights(self, json_file, weights_file):
        if os.path.exists(json_file) and os.path.exists(weights_file):
            # Model reconstruction from JSON file
            with open(json_file, 'r') as f:
                self.m_model = model_from_json(f.read())

            # Load weights into the new model
            self.m_model.load_weights(weights_file)

    def save_json_weights(self, path="./model/", namefile="facerec"):
        """Export trained model split in two files: the scheme model in format 'json' and weights'file in format h5."""
        if not os.path.exists(path):
            os.makedirs(path)
        # print the scheme model
        plot_model(self.m_model, to_file='model.png')
        # save as JSON
        model_json = self.m_model.to_json()
        with open(path + namefile + ".json", "w") as json_file:
            json_file.write(model_json)
        # save weights
        self.m_model.save_weights(path + namefile + '_weights.h5')
        # print the scheme model
        plot_model(self.m_model, to_file='model.png')

    def save_model(self, path="./model/", namefile="facerecognition"):
        """Export trained model in format '.model'."""
        if not os.path.exists(path):
            os.makedirs(path)
        self.m_model.save(path + namefile + ".model")
        # print the scheme model
        plot_model(self.m_model, to_file='model.png')

    def face_recognition_model(self):
        """Model face recognition based on model VGG16 Oxford University."""
        # define input model block
        x_input = Input((3, self.m_image_width, self.m_image_height))
        x = (ZeroPadding2D((1, 1)))(x_input)
        # block 1
        x = (Convolution2D(64, (3, 3), activation='relu', padding="same", name="block1_conv1"))(x)
        x = (ZeroPadding2D((1, 1)))(x)
        x = (Convolution2D(64, (3, 3), activation='relu', padding="same", name="block1_conv2"))(x)
        x = (MaxPooling2D((2, 2), data_format="channels_first", strides=(2, 2)))(x)
        # block 2
        x = (ZeroPadding2D((1, 1)))(x)
        x = (Convolution2D(128, (3, 3), activation='relu', padding="same", name="block2_conv1"))(x)
        x = (ZeroPadding2D((1, 1)))(x)
        x = (Convolution2D(128, (3, 3), activation='relu', padding="same", name="block2_conv2"))(x)
        x = (MaxPooling2D((2, 2), data_format="channels_first", strides=(2, 2)))(x)
        # block 3
        x = (ZeroPadding2D((1, 1)))(x)
        x = (Convolution2D(256, (3, 3), activation='relu', padding="same", name="block3_conv1"))(x)
        x = (ZeroPadding2D((1, 1)))(x)
        x = (Convolution2D(256, (3, 3), activation='relu', padding="same", name="block3_conv2"))(x)
        x = (ZeroPadding2D((1, 1)))(x)
        x = (Convolution2D(256, (3, 3), activation='relu', padding="same", name="block3_conv3"))(x)
        x = (MaxPooling2D((2, 2), data_format="channels_first", strides=(2, 2)))(x)
        # block 4
        x = (ZeroPadding2D((1, 1)))(x)
        x = (Convolution2D(512, (3, 3), activation='relu', padding="same", name="block4_conv1"))(x)
        x = (ZeroPadding2D((1, 1)))(x)
        x = (Convolution2D(512, (3, 3), activation='relu', padding="same", name="block4_conv2"))(x)
        x = (ZeroPadding2D((1, 1)))(x)
        x = (Convolution2D(512, (3, 3), activation='relu', padding="same", name="block4_conv3"))(x)
        x = (MaxPooling2D((2, 2), data_format="channels_first", strides=(2, 2)))(x)
        # block 5
        x = (ZeroPadding2D((1, 1)))(x)
        x = (Convolution2D(512, (3, 3), activation='relu', padding="same", name="block5_conv1"))(x)
        x = (ZeroPadding2D((1, 1)))(x)
        x = (Convolution2D(512, (3, 3), activation='relu', padding="same", name="block5_conv2"))(x)
        x = (ZeroPadding2D((1, 1)))(x)
        x = (Convolution2D(512, (3, 3), activation='relu', padding="same", name="block5_conv3"))(x)
        x = (MaxPooling2D((2, 2), data_format="channels_first", strides=(2, 2)))(x)
        # classification block
        x = (Convolution2D(4096, (7, 7), activation='relu', name="fc1"))(x)
        x = (Dropout(0.5))(x)
        x = (Convolution2D(4096, (1, 1), activation='relu', name="fc2"))(x)
        x = (Dropout(0.5))(x)
        x = (Convolution2D(2622, (1, 1)))(x)
        x = (Flatten())(x)
        x = (Activation('softmax'))(x)
        # output layer - predictions
        predictions = (Dense(self.m_num_classes, activation='softmax', name="predictions"))(x)
        # create model instance
        self.m_model = Model(inputs=x_input, outputs=predictions, name="FaceRecognitionModelVGG16")
        # compile the  model
        self.m_model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                             loss='categorical_crossentropy', metrics=['accuracy'])

        # print model structure diagram
        print(self.m_model.summary())


if __name__ == '__main__':
    test = FaceRecognition(epochs=1, batch_size=32, image_width=48, image_height=48)
    test.create_img_generator()
    test.set_train_generator(
        train_folder=r'./dataset/simpsons_dataset')
    test.set_valid_generator(
        validate_folder=r'./dataset/simpsons_dataset')
    test.set_test_generator(
        test_folder=r'./dataset/kaggle_simpson_testset')
    # test.input_model(r'/Users/francesco/Downloads/the-simpsons-characters-dataset/weights.best.hdf5')
    test.train_and_fit_model()
    test.predict_class_indices()
    test.predict_output()
    test.save_model()
    quit(0)
