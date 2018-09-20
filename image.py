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
    _num_classes = 0

    def __init__(self, dataset, labels, image_width=299, image_height=299):
        self._dataset = dataset
        self._image_width, self._image_height = image_width, image_height
        if labels is dict:
            self._num_classes = len(labels)
        else:
            assert "Wrong Type of the second parameter, it must be a dict"

    def prepare_pictures(self):
        """
        Load pictures from folders for characters from the map_characters dict and create a numpy dataset and
        a numpy labels set. Pictures are re-sized into picture_size square.
        :param BGR: boolean to use true color for the picture (RGB instead of BGR for plt)
        :return: dataset, labels set
        """
        pics = []
        labels = []
        for i, pictures in self._num_classes.items():
            pict_sel = [sel for sel in self._dataset if pictures in sel]
            for images in pict_sel:
                # for k, char in mapped.items():
                # pictures = [k for k in glob.glob('./characters/%s/*' % char)]
                # nb_pic = round(pictures_per_class / (1 - test_size)) if round(pictures_per_class / (1 - test_size)) < len(
                #   pictures) else len(pictures)
                # nb_pic = len(pictures)
                # for pic in np.random.choice(pictures, nb_pic):
                tmp_img = cv2.imread(images)
                tmp_img = cv2.resize(tmp_img, (self._image_width, self._image_height))
                pics.append(tmp_img)
                labels.append(i)

        # store array image and labels not normalize
        self._X, self._y = np.array(pics), np.array(labels, dtype='int64')
        # normalize value with 255
        X_Train = self._X.astype('float32') / self._normalize
        y_train = keras.utils.to_categorical(self._y, self._num_classes)

        # return np.array(pics), np.array(labels)
        return X_Train, y_train


if __name__ == '__main__':
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    from preparedata import DataSet

    path_datest1 = '/Users/francesco/Downloads/the-simpsons-characters-dataset/simpsons_dataset/'
    dataset = DataSet(path_datest1)

    img_data, labels = dataset.get_dataset()
    sublables = {k: v for k, v in labels.items() if k in range(0,4,1)}


    img = PreparePicture(dataset, sublables)
    #X_train, y_train
    img.prepare_pictures()

    # img_data_list, labels =
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
    ########    ###########################################################################################################

