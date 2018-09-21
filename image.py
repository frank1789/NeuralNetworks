#!usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import keras
import numpy as np


class PreparePicture(object):
    """Class to handle and edit images in the data set and configure according to the labels returns the number of
    classes you want to classify
    """
    _X_train = None
    _y_train = None
    _normalize = 255.0

    def __init__(self, input_dataset, input_labels, image_width=299, image_height=299):
        """Default constructor, pass in argument the file's data-set and previous label-set, then set (optionally) the
        dimension of image the default factor is square (299, 299)
        :param input_dataset (list) data-set
        :param input_labels (dict) labels of data-set from subfolder
        :param image_width (int)  specify width dimension
        :param image_height (int) specify height dimension
        """
        self._dataset = input_dataset
        self._image_width, self._image_height = image_width, image_height
        if type(input_labels) is dict:
            self._labels = input_labels
            self._num_label = len(input_labels)
        else:
            assert "Wrong Type of the second parameter, it must be a dict"

    def prepare_pictures(self):
        """
        Load pictures from folders for characters from the map_characters dict and create a numpy data-set and
        a numpy labels set. Pictures are re-sized into picture_size defined in constructor.
        """
        pics_ = []
        labels_ = []
        for i, pict_labels in self._labels.items():
            # filter image not in labels, then it's possible use a restricted data-set
            images = [select for select in self._dataset if pict_labels in select]
            for img in images:
                tmp_img = cv2.imread(img)
                tmp_img = cv2.resize(tmp_img, (self._image_width, self._image_height))
                pics_.append(tmp_img)
                labels_.append(i)
        X = np.array(pics_)  # array image as numpy-array
        y = np.array(labels_, dtype='int64')  # array labels as numpy-array
        # normalize value with 255
        self._X_train = X.astype('float32') / self._normalize
        self._y_train = keras.utils.to_categorical(y, self._num_label)

    def get_image(self):
        """ Return entire data_set and label data-set as numpy-array normalized.
        :return: (numpy-array) data-set, labels-set
        """
        # return np.array(pics), np.array(labels)
        return self._X_train, self._y_train

    def get_num_class(self):
        """Return number of class analyzed from label data-set.
        :return (int) number of class"""
        return self._num_label


if __name__ == '__main__':
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    from preparedata import DataSet

    path_datest1 = '/Users/francesco/Downloads/the-simpsons-characters-dataset/simpsons_dataset/'
    dataset = DataSet(path_datest1)

    img_data, labels = dataset.get_dataset()
    sublables = {k: v for k, v in labels.items() if k in range(0, 4, 1)}
    print(labels.items())
    img = PreparePicture(img_data, sublables, 32, 32)
    img.prepare_pictures()
    num_classes = img.get_num_class()

    X, y = img.get_image()
