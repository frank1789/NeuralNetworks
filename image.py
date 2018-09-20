import os
import cv2
import keras
import numpy as np


class PreparePicture(object):
    """


    """
    _X = None
    _y = None
    _normalize = 255.0


    def __init__(self, dataset, labels, image_width=299, image_height=299):
        """Default constructor, pass in argument the file's data-set and previous label-set, then set (optionally) the
        dimension of image the default factor is square (299, 299)
        :param dataset (list) data-set
        :param labels (dict) labels of data-set from subfolder
        :param image_width (int)  specify width dimension
        :param image_height (int) specify height dimension
        """
        self._dataset = dataset
        self._image_width, self._image_height = image_width, image_height
        if type(labels) is dict:
            self._labels = labels
            self._num_label = len(labels)
        else:
            assert "Wrong Type of the second parameter, it must be a dict"

    def prepare_pictures(self):
        """
        Load pictures from folders for characters from the map_characters dict and create a numpy data-set and
        a numpy labels set. Pictures are re-sized into picture_size square.
        :return: (numpy-array) data-set, labels-set
        """
        pics = []
        labels = []
        #print(self._labels)
        for i, pict_labels in self._labels.items():
            images = [select for select in self._dataset if pict_labels in select]
            for img in images:
                tmp_img = cv2.imread(img)
                tmp_img = cv2.resize(tmp_img, (self._image_width, self._image_height))
                pics.append(tmp_img)
                labels.append(i)

        # store array image and labels not normalize
        self._X, self._y = np.array(pics), np.array(labels, dtype='int64')
        # normalize value with 255
        X_Train = self._X.astype('float32') / self._normalize
        y_train = keras.utils.to_categorical(self._y, self._num_label)

        # return np.array(pics), np.array(labels)
        return X_Train, y_train


if __name__ == '__main__':
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    from preparedata import DataSet

    path_datest1 = '/Users/francesco/Downloads/the-simpsons-characters-dataset/simpsons_dataset/'
    dataset = DataSet(path_datest1)

    img_data, labels = dataset.get_dataset()
    sublables = {k: v for k, v in labels.items() if k in range(0, 4, 1)}
    print(labels.items())
    img = PreparePicture(img_data, sublables)

    X_train, y_train = img.prepare_pictures()
