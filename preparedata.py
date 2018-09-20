#!usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import re
import fnmatch
import cv2
import numpy as np
import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

# Saving and loading model and weights
from keras.models import model_from_json
from keras.models import load_model

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


class PrepareDataset(object):
    """ It is a class that allows to prepare the analysis of the dataset once imported from a source.
    Inside there are methods for manipulating the data-set.
    """
    defalut_dataset_folder = './data/dataset'  # folder contains the data-set and his sub-folder
    defalut_datatest_folder = './data/test'  # folder contains test-set

    _exclude_ext = ['.DS_Store']  # extensions of the files to be ignored, such as hidden files
    __data_category = None  # collect name of sub-folder in data-set

    def __init__(self):
        if not os.path.exists(self.defalut_dataset_folder):
            os.makedirs(self.defalut_dataset_folder)
            print("Local 'dataset' folder created")

        if not os.path.exists(self.defalut_datatest_folder):
            os.makedirs(self.defalut_datatest_folder)
            print("Local 'testset' folder created")

    def _scan_folder(self, path):
        """Scans the folder by discarding all files that have an unsupported extension.
        :param path (str) folder's path
        :return list_files (list) list contains files
        """
        list_files = []
        if os.path.exists(path):  # check if is valid path
            for root, dirname, files in os.walk(path):
                files = [f for f in files if not re.match(self.__exclude_file(), f)]  # filter files
                for namefile in files:
                    list_files.append(os.path.join(root, namefile))  # fill the lst with all files

        return list_files

    def _build_dataset(self, path):
        """Generates a dict where it creates a numeric reference for each sub-folder useful for selecting sub-groups of
        the data-set.
        :param path_dataset (str) data-set's path
        """
        list_files = self._scan_folder(path)
        # compile category list from sub-folder of data-set
        directory_name = os.listdir(path)
        directory_name = [f for f in directory_name if not re.match(self.__exclude_file(), f)]  # filter files
        directory_name.sort(reverse=False)
        list_category = directory_name
        # return files' data-set and category
        self.__data_category = list_category
        return list_files

    def set_exclude_file(self, exclude):
        """Allows you to add to the list of extensions those you want to ignore by passing them as 'list' or 'str'
        :param exclude (str, list) extension's file
        """
        if type(exclude) is list:
            for i in exclude:
                self._exclude_ext.append(i)
        else:
            self._exclude_ext.append(exclude)

    def __exclude_file(self):
        """Transform glob/os.walk in str and use regex to compare"""
        excludes = r'|'.join([fnmatch.translate(x) for x in self._exclude_ext]) or r'$.'
        return excludes

    def _get_map_category(self):
        """Generate a classifiction of sub-folder used to sub-selection"""
        temp_list = []
        for i in range(0, len(self.__data_category), 1):
            temp_list.append(i)

        return dict(zip(temp_list, self.__data_category))

    def copy_file(self):
        pass


class DataSet(PrepareDataset):
    """Class manage the data-set."""
    __list_files = []  # list of files in data-set
    __list_category = None  # data-set categories

    def __init__(self, path_dataset):
        """Default constructor of data-set. Once invoked it checks the validity of the examined folder, proceeds in the
        generation of the list of the files of the dataset and of the relative classification.
        :param path_dataset (str) path's data-set.
        """
        super(DataSet, self).__init__()
        # check if folder exist, then build the database
        if os.path.exists(path_dataset):
            self.__list_files = self._build_dataset(path_dataset)
            # make the category list
            self.__list_category = self._get_map_category()

    def get_dataset(self):
        """Return the data-set and sub-category.
        :returns (list) files
        :return (dict) category's data-set"""
        return self.__list_files, self.__list_category

    def copy_file(self):
        """Copy from original."""
        for file in self.__list_files:
            shutil.copy(file, self.defalut_dataset_folder)


class TestSet(PrepareDataset):
    """Class manage the Test set"""
    __list_files = []  # list of files in test-set

    def __init__(self, path_testset):
        """Default constructor of data-set. Once invoked it checks the validity of the examined folder, proceeds in the
        generation of the list of the files to validate the model.
        :param path_testset (str) path's test-set.
        """
        super(TestSet, self).__init__()
        if os.path.exists(path_testset):
            self.__list_files = self._scan_folder(path_testset)
            self.copy_file()

    def copy_file(self):
        for file in self.__list_files:
            shutil.copy(file, self.defalut_datatest_folder)





if __name__ == '__main__':
    # test 1
    datest1 = '/Users/francesco/Downloads/the-simpsons-characters-dataset/simpsons_dataset/'
    test = '/Users/francesco/Downloads/the-simpsons-characters-dataset/kaggle_simpson_testset'
    d = DataSet(datest1)
    d.copy_file()
    t = TestSet(test)
    t.copy_file()
    t.set_exclude_file(['.txt', '.csv'])
    dataset, category = d.get_dataset()
    print(dataset)
    print(category)
    t.set_exclude_file('.tiff')
    # d.build_dataset()
    # test 2
    # d, labels = PrepareDataset('/Users/francesco/PycharmProjects/KerasTest/data/train').get_map_character()
    # print(d, labels)
    # quit()
    # load and prepare data
    people = ['Cristiano_Ronaldo', 'Jackie_Chan', 'Lionel_Messi', 'Rathanak']
    num_classes = 4
    img_data_list = []
    labels = []
    valid_images = [".jpg", ".gif", ".png"]
    quit()
    # print(d.get_dataset())

    # for index, person in enumerate(people):
    #     print(index)
    #     dir_path = 'images/' + person
    #     print(dir_path)
    #     #for img_path in os.listdir(dir_path):
    #     #    name, ext = os.path.splitext(img_path)
    #     #    if ext.lower() not in valid_images:
    #     #        continue
    # for i, img in enumerate(d.get_dataset()):
    #     # print(img)
    #     img_data = cv2.imread(img)
    #     # cv2.imshow('Image', img_data)
    #     # cv2.waitKey(0)
    #     # cv2.destroyAllWindows()
    #     # convert image to gray
    #     img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
    #     # cv2.imshow('Image', img_data)
    #     # cv2.waitKey(0)
    #     # cv2.destroyAllWindows()
    #     img_data_list.append(img_data)
    #     labels.append(i)

    # map1 = d.get_map_category()
    # subdict = {k: v for k, v in map1.items() if k in range(0,4,1)}
    # print(subdict)
    #
    #
    # img_data_list, labels = PreparePicture(d.get_dataset()).prepare_pictures(subdict)
    # print(labels)
    #
    # img_data = np.array(img_data_list)
    # img_data = img_data.astype('float32')
    #
    # labels = np.array(labels, dtype='int64')
    # print(labels)
    # # scale down(so easy to work with)
    # img_data /= 255.0
    # img_data = np.expand_dims(img_data, axis=4)
    # print (img_data.shape)
    # print (img_data.shape[0])
    # print(img_data.shape)
    # print(labels.shape)

    # convert class labels to on-hot encoding
    # Y = np_utils.to_categorical(labels, num_classes)
    # print(Y)
    ########    ############################################################################################################

    X, y = PreparePicture(d.get_dataset()).prepare_pictures(subdict)

    print(X[0].shape)

    input_shape = X.shape[1:]

    model = Sequential()
    model.add(
        Conv2D(32, (3, 3), input_shape=input_shape, padding='same', activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile model
    epochs = 2
    lrate = 0.01
    decay = lrate / epochs
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    print(model.summary())

    # Fit the model
    model.fit(X, y, epochs=epochs, batch_size=32)

    # Final evaluation of the model
    # scores = model.evaluate(X_test, y_test, verbose=0)
    # print("Accuracy: %.2f%%" % (scores[1] * 100))
