#!usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from utilityfunction import Spinner
import errno
import tensorflow as tf
from keras.models import Model, model_from_json, load_model
from keras.layers import Dropout, Flatten, Dense, Input
from keras.optimizers import SGD, Adam
from keras.layers.convolutional import ZeroPadding2D
from keras.layers import MaxPooling2D, Convolution2D, Activation, GlobalAveragePooling2D
from keras.applications import VGG16, VGG19, InceptionV3, Xception, ResNet50
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from keras import backend as kbe

# from staticsanalysis import HistoryAnalysis

# suppress warning and error message tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# specify input shape
kbe.set_image_dim_ordering('tf')


class FaceRecognition(object):
    m_train_generator = None
    m_valid_generator = None
    m_model = None
    m_num_train_samples = 0
    m_num_classes = 0
    m_num_validate_samples = 0
    m_model_base_ = None

    def __init__(self, epochs, batch_size, learning_rate=0.0001, image_width=224, image_height=224):
        self.m_epochs = epochs
        self.m_batch_size = batch_size
        self.m_lr = learning_rate
        self.m_image_width = image_width
        self.m_image_height = image_height
        self.__pathdir = 'Model'
        self.__spin = Spinner()

    @staticmethod
    def get_num_files(path):
        """
        Get count of number of files in this folder and all sub-folders
        :param path: (str) path folder
        :return: (int)
        """
        if not os.path.exists(path):
            return 0
        return sum([len(files) for r, d, files in os.walk(path)])

    @staticmethod
    def get_num_subfolders(path):
        """
        Get count of number of sub-folders directly below the folder in path.
        :param path: (str) path folder
        :return: (int)
        """
        if not os.path.exists(path):
            return 0
        return sum([len(d) for r, d, files in os.walk(path)])

    @staticmethod
    def create_img_generator(rotation_range=30, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,
                             zoom_range=0.2, horizontal_flip=True):
        """
        Define image generators that will variations of image with the image rotated slightly, shifted up, down,
        left, or right, sheared, zoomed in, or flipped horizontally on the vertical axis (ie. person looking to the
        left ends up looking to the right).
        :param rotation_range: (int) generates a rotation of the image of the specify value
        :param width_shift_range: (float) generates a displacement of the image of the specify value
        :param height_shift_range: (float) generates a displacement of the image of the specify value
        :param shear_range: (float) generates a cut-out of the image of the specify value
        :param zoom_range: (float) generates a magnification of the image of the specify value
        :param horizontal_flip: (bool) enable flips the image horizontally
        :return ImageDataGenerator: (object)
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

    def set_train_generator(self, train_dir):
        """
        Connect the image generator to a folder contains the source images the image generator alters.
        :param train_dir: (str) containing the path to the folder
        """
        #   Training image generator
        self.m_num_train_samples = self.get_num_files(train_dir)
        self.m_num_classes = self.get_num_subfolders(train_dir)
        train_generator = self.create_img_generator()
        self.m_train_generator = train_generator.flow_from_directory(
            directory=train_dir,
            target_size=(self.m_image_width, self.m_image_height),
            batch_size=self.m_batch_size,
            seed=42  # set seed for reproducibility
        )

    def set_valid_generator(self, valid_dir):
        """
        Connect the image generator to a folder contains the source images the image generator alters.
        :param valid_dir: (str) containing the path folder
        """
        self.m_num_validate_samples = self.get_num_files(valid_dir)
        valid_generator = self.create_img_generator()
        self.m_valid_generator = valid_generator.flow_from_directory(
            directory=valid_dir,
            target_size=(self.m_image_width, self.m_image_height),
            batch_size=self.m_batch_size,
            seed=42  # set seed for reproducibility
        )

    def train_and_fit_model(self, figure_history_name):
        """Train the model"""
        # Fit
        history = self.m_model.fit_generator(
            self.m_train_generator,
            epochs=self.m_epochs,
            steps_per_epoch=self.m_num_train_samples // self.m_batch_size,
            validation_data=self.m_valid_generator,
            validation_steps=self.m_num_validate_samples // self.m_batch_size,
            shuffle=True,
            class_weight='auto', )
        # print plot training (accuracy vs lost)

    # plotter = HistoryAnalysis.plot_history(history, figure_history_name)

    def load_model_from_file(self, filename, weights_file=None):
        """
        Import trained model store as 1 file ('.model', '.h5')
        Or import the schema model in format 'json' and weights's file in format h5.
        :param filename: (str) pass path model file
        :param weights_file: (str) pass path weights file
        """
        if os.path.exists(filename) and weights_file is None:
            print("Loading model, please wait")
            self.__spin.start()
            # load entire model
            if filename.endswith(('.model', '.h5')):
                self.m_model = load_model(filename)
                self.__spin.stop()
                print("Done")
            else:
                self.__spin.stop()
                raise ValueError("Invalid extension, supported extensions are: '.h5', '.model'")

        elif os.path.exists(filename) and weights_file is not None:
            if filename.endswith('.json') and weights_file.endswith('.h5'):
                print("Loading model, please wait")
                self.__spin.start()
                # Model reconstruction from JSON file
                with open(filename, 'r') as f:
                    self.m_model = model_from_json(f.read())
                # Load weights into the new model
                self.m_model.load_weights(weights_file)
                self.__spin.stop()
                print("Done")
            else:
                self.__spin.stop()
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), (filename, weights_file))

        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)

        print(self.m_model.summary())

    def save_model_to_file(self, name='model', extension='.h5', export_image=False):
        """
        Export trained model store as 1 file ('.model', '.h5')
        or export the schema model in format 'json' and weights's file in format h5.
        :param name: (str) assign file name
        :param extension: (str) assign file extension
        :param export_image: (bool) generate figure schema model
        """
        print("Saving model, please wait")
        # self.__spin.start()
        if not os.path.exists(self.__pathdir):
            os.makedirs(self.__pathdir)
        # build the name-file
        filename = os.path.join(self.__pathdir, (name + extension))
        if extension not in ['.h5', '.model', '.json']:
            self.__spin.stop()
            raise ValueError("Invalid extension, supported extensions are: '.h5', '.model', '.json'")
        if extension == '.h5':
            # export complete model with weights
            self.m_model.save(filename)
        elif extension == '.model':
            # export complete model with weights
            self.m_model.save(filename)
        elif extension == '.json':
            # save as JSON and weights
            model_json = self.m_model.to_json()
            with open(filename, "w") as json_file:
                json_file.write(model_json)
            # save weights
            namefile_weights = os.path.join(os.path.join(self.__pathdir, (name + '_weights.h5')))
            print(namefile_weights)
            self.m_model.save_weights(namefile_weights)

        if export_image:
            image_name = os.path.join(self.__pathdir, (name + '.png'))
            # print image of schema model
            plot_model(self.m_model, to_file=image_name, show_layer_names=True, show_shapes=True)

        # self.__spin.stop()
        print("Done")

    def get_pretrained_model(self, pretrained_model, weights='imagenet', include_top=False):
        """
        This method allows to define already existing neural networks and to export their model as a starting point.
        :param pretrained_model: (string) set name of neural network as inception, vgg16 ecc.
        :param weights: (string) set weights of NN
        :param include_top: (bool)
        :return model_base: (obj) the model select
        :return output: (obj) pre-trained NN weights
        """
        if pretrained_model == 'inception':
            model_base = InceptionV3(include_top=include_top,
                                     weights=weights,
                                     input_shape=self.m_train_generator.image_shape)
            output = model_base.output

        elif pretrained_model == 'xception':
            model_base = Xception(include_top=include_top,
                                  weights=weights,
                                  input_shape=self.m_train_generator.image_shape)
            output = (Flatten())(model_base.output)

        elif pretrained_model == 'resnet50':
            model_base = ResNet50(include_top=include_top,
                                  weights=weights,
                                  input_shape=self.m_train_generator.image_shape)
            output = model_base.output

        elif pretrained_model == 'vgg16':
            model_base = VGG16(include_top=include_top,
                               weights=weights,
                               input_shape=self.m_train_generator.image_shape)
            output = model_base.output

        elif pretrained_model == 'vgg19':
            model_base = VGG19(include_top=include_top,
                               weights=weights,
                               input_shape=self.m_train_generator.image_shape)
            output = model_base.output

        return model_base, output

    def set_face_recognition_model(self, pretrained_model='', weights='', include_top=False, Number_FC_Neurons=1024,
                                   trainable_parameters=False, num_trainable_parameters=1.0):
        """
        It allows to use different neural networks of convulsion, the first is based on the face recognition model
        VGG16 Oxford University, available at the address '' and personalized.
        Instead, those defined succinctly are the expansion of existing networks with the addition of the custom
        classifier. Furthermore, it is possible to specify whether to train all parameters or just some parameters,
        simply setting a dimensionless range between 0 and 1.0.
        :param pretrained_model:
        :param weights:
        :param include_top:
        :param Number_FC_Neurons:
        :param trainable_parameters:
        :param num_trainable_parameters:
        :return self(object):
        """
        if pretrained_model == '':  # use own custom model
            try:  # check minimum size image
                # define input model block
                x_input = Input(self.m_train_generator.image_shape)
                self.m_model_base_ = x_input
                x = (ZeroPadding2D((1, 1), name="InputLayer"))(x_input)
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
                x = (Convolution2D(256, (3, 3), activation='relu', padding="same", name="block3_conv1x"))(x)
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
                x = (Convolution2D(4096, (7, 7), activation='relu', padding="same", name="fc1"))(x)
                x = (Dropout(0.5))(x)
                x = (Convolution2D(4096, (1, 1), activation='relu', name="fc2"))(x)
                x = (Dropout(0.5))(x)
                x = (Convolution2D(2048, (1, 1)))(x)
                # Classification block
                x = Flatten()(x)
                x = Activation(activation='softmax')(x)

            except ValueError as err:
                message = "ValueError:Input size must be at least 48 x 48;"
                message += " got `input_shape=" + str(self.m_train_generator.image_shape) + "'"
                print(message)
                raise err

        elif pretrained_model == 'inception':
            model_base, output = self.get_pretrained_model(pretrained_model, weights, include_top)
            self.m_model_base_ = model_base.input
            x = GlobalAveragePooling2D()(output)
            x = Dense(Number_FC_Neurons, activation='relu')(x)  # new FC layer, random init

        elif pretrained_model == 'xception':
            model_base, output = self.get_pretrained_model(pretrained_model, weights, include_top)
            self.m_model_base_ = model_base.input
            # classification block
            x = Dense(Number_FC_Neurons, activation='relu')(output)  # new FC layer, random init

        elif pretrained_model == 'resnet50':
            model_base, output = self.get_pretrained_model(pretrained_model, weights, include_top)
            self.m_model_base_ = model_base.input
            x = Flatten(name='flatten')(output)
            x = Dropout(0.5)(x)

        elif pretrained_model == 'vgg16' or pretrained_model == 'vgg19':
            model_base, output = self.get_pretrained_model(pretrained_model, weights, include_top)
            self.m_model_base_ = model_base.input
            # classification block
            x = Flatten()(output)
            x = Dense(Number_FC_Neurons, activation='relu')(x)
            x = Dropout(0.5)(x)
            x = Dense(Number_FC_Neurons, activation='relu')(x)
            x = Dropout(0.5)(x)

        else:
            print("Neural network not available")
            sys.exit()

        # common output layer - predictions
        predictions = Dense(self.m_num_classes, activation='softmax', name="predictions")(x)
        # create model instance
        self.m_model = Model(inputs=self.m_model_base_, outputs=predictions)
        # Layers - set trainable parameters
        print("Total layers: {:10d}".format(len(self.m_model.layers)))
        if trainable_parameters:
            if 0 < num_trainable_parameters < 1.0:
                layers2freeze = int(len(self.m_model.layers) * num_trainable_parameters) + 1
                for layer in self.m_model.layers[:layers2freeze]:
                    layer.trainable = False
                for layer in self.m_model.layers[layers2freeze:]:
                    layer.trainable = True
            else:
                for layer in self.m_model.layers:
                    layer.trainable = False

        # compile the  model
        if self.m_num_classes == 2:
            self.m_model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                                 loss='categorical_crossentropy', metrics=['accuracy'])
        else:
            self.m_model.compile(optimizer='adam',
                                 loss='categorical_crossentropy', metrics=['accuracy'])

        # print model structure diagram
        print(self.m_model.summary())
        return self

    def __del__(self):
        del self.m_model
        del self.m_model_base_
        del self.m_train_generator
        del self.__spin


if __name__ == '__main__':
    train_folder = r'/Users/francesco/PycharmProjects/KerasTest/data/train'
    valid_folder = r'/Users/francesco/PycharmProjects/KerasTest/data/validate'
    test = FaceRecognition(epochs=10, batch_size=32, image_width=48, image_height=48)
    test.create_img_generator()
    test.set_train_generator(
        train_dir=train_folder)
    test.set_valid_generator(
        valid_dir=valid_folder)

    # prepare the model
    test.set_face_recognition_model(pretrained_model='vgg16', weights='imagenet', include_top=False)

    # train fit
    test.train_and_fit_model()
    quit(0)
