#!usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import re
import fnmatch
import random


class PrepareDataset(object):
    """
    It is a class that allows to prepare the analysis of the dataset once imported from a source.
    Inside there are methods for manipulating the data-set.
    """

    _default_train = r'./data/train'  # folder contains the data-set and his sub-folder
    _default_validate = r'./data/validate'  # folder contains the data-set and his sub-folder
    _default_test = r'./data/test'  # folder contains test-set

    _exclude_ext = ['.DS_Store', 'desktop.ini', 'Desktop.ini', '.csv', '.json',
                    '.h5']  # extensions of the files to be ignored, such as hidden files
    __data_category = None  # collect name of sub-folder in data-set

    def __init__(self):
        if not os.path.exists(self._default_train):
            # make train folder
            os.makedirs(self._default_train)
            print("Genarated train folder created at: ", self._default_train)
        if not os.path.exists(self._default_validate):
            # make validate folder
            os.makedirs(self._default_validate)
            print("Generated train folder created at: ", self._default_validate)
        if not os.path.exists(self._default_test):
            # make test foler
            os.makedirs(self._default_test)
            print("Generate test folder created at: ", self._default_test)

    def _scan_folder(self, path):
        """
        Scans the folder by discarding all files that have an unsupported extension.
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
        """
        Generates a dict where it creates a numeric reference for each sub-folder useful for selecting sub-groups of
        the data-set.
        :param path (str) data-set's path
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
        """
        Allows you to add to the list of extensions those you want to ignore by passing them as 'list' or 'str'
        :param exclude (str, list) extension's file
        """
        if type(exclude) is list:
            for i in exclude:
                self._exclude_ext.append(i)
        else:
            self._exclude_ext.append(exclude)

    def __exclude_file(self):
        """
        Transform glob/os.walk in str and use regex to compare
        """
        excludes = r'|'.join([fnmatch.translate(x) for x in self._exclude_ext]) or r'$.'
        return excludes

    def _get_map_category(self):
        """Generate a classifiction of sub-folder used to sub-selection"""
        temp_list = []
        for i in range(0, len(self.__data_category), 1):
            temp_list.append(i)

        return dict(zip(temp_list, self.__data_category))
    #
    # @staticmethod
    # def walkdir(folder):
    #     """Walk through each files in a directory"""
    #     for dirpath, dirs, files in os.walk(folder):
    #         for filename in files:
    #             yield os.path.abspath(os.path.join(dirpath, filename))
    #
    # def dosomething(self, dir, outputdir, train_split_validate):
    #     files = [file for file in os.listdir(dir) if os.path.isfile(os.path.join(dir, file))]
    #     files.sort()
    #     # Amount of random files you'd like to select
    #     random_amount = int((len(files) * train_split_validate) + 1)
    #     for x in range(random_amount):
    #         if len(files) == 0:
    #             break
    #         else:
    #             file = random.choice(files)
    #             shutil.copy(os.path.join(dir, file), outputdir)
    #
    # def process_content_with_progress3(self, inputpath, outputdir, split_train_validate, blocksize=1024):
    #     # Preprocess the total files sizes
    #     sizecounter = 0
    #     for filepath in tqdm(self.walkdir(inputpath), unit="files"):
    #         sizecounter += os.stat(filepath).st_size
    #
    #     # Load tqdm with size counter instead of file counter
    #     with tqdm(total=sizecounter,
    #               unit='B', unit_scale=True, unit_divisor=1024) as pbar:
    #         for filepath in self.walkdir(inputpath):
    #             with open(filepath, 'rb') as fh:
    #                 buf = 1
    #                 while (buf):
    #                     buf = fh.read(blocksize)
    #                     self.dosomething(inputpath, self._default_validate, split_train_validate)
    #                     if buf:
    #                         pbar.set_postfix(file=filepath[-10:], refresh=False)
    #                         pbar.update(len(buf))


class DataSet(PrepareDataset):
    """Class manage the data-set."""
    __list_files = []  # list of files in data-set
    __list_category = None  # data-set categories
    _subdirs = []

    def __init__(self, raw_dataset):
        """
        Default constructor of data-set. Once invoked it checks the validity of the examined folder, proceeds in the
        generation of the list of the files of the dataset and of the relative classification.
        :param path_dataset (str) path's data-set.
        """
        super(DataSet, self).__init__()
        # check if folder exist, then build the database
        try:
            if os.path.exists(raw_dataset):
                for subdirs in os.listdir(raw_dataset):
                    if subdirs not in self._exclude_ext:
                        self._subdirs.append(os.path.join(raw_dataset, subdirs))
                    else:
                        pass
        except OSError:
            if not os.path.isdir(raw_dataset):
                raise
        self._subdirs.sort()

    def get_dataset(self):
        """
        Return the data-set and sub-category.
        :returns (list) files
        :return (dict) category's data-set"""
        return self.__list_files, self.__list_category

    def copy_file(self, split_train_validate=30):
        """
        Copy from original folder and split in two folder train and validate
                            data
                             |
                    +--------+--------+
                    |                 |
                  train            validate
        """
        try:
            if 0 < split_train_validate <= 100:
                split_train_validate /= 100
                for raw in self._subdirs:
                    for root, dir, files in os.walk(raw):
                        files.sort()
                        # Amount of random files you'd like to select
                        random_amount = int((len(files) * split_train_validate) + 1)
                        mess = "Total files in {:8s},".format(root)
                        mess += "\tsplit to {:3d}%,".format(int(split_train_validate * 100))
                        mess += "\tfiles in validate folder: {:5d}".format(random_amount)
                        print(mess)
                        for x in range(random_amount):
                            if len(files) == 0:
                                break
                            else:
                                file = random.choice(files)
                                tmpdir = os.path.split(root)[1]
                                dest = os.path.join(self._default_validate, tmpdir)
                                if not os.path.exists(dest):
                                    os.makedirs(dest)
                                else:
                                    pass
                                shutil.copy(os.path.join(root, file), dest)
        except ValueError:
            print(ValueError, "train_split_validate must be a number to divide the training set must be 0 and 100")


# class TestSet(PrepareDataset):
#     """
#     Class manage the Test set
#     """
#     __list_files = []  # list of files in test-set
#
#     def __init__(self, path_testset):
#         """
#         Default constructor of data-set. Once invoked it checks the validity of the examined folder, proceeds in the
#         generation of the list of the files to validate the model.
#         :param path_testset (str) path's test-set.
#         """
#         super(TestSet, self).__init__()
#         if os.path.exists(path_testset):
#             self.__list_files = self._scan_folder(path_testset)
#             self.copy_file()
#
#     def copy_file(self):
#         pass
#         # for file in self.__list_files:
#         #    shutil.copy(file, self.defalut_datatest_folder)
#
#     def __del__(self):
#         pass


if __name__ == '__main__':
    # test 1
    datest1 = '/Users/francesco/Downloads/the-simpsons-characters-dataset/simpsons_dataset/'
    test = '/Users/francesco/Downloads/the-simpsons-characters-dataset/kaggle_simpson_testset'

    d = DataSet(datest1)
    d.copy_file(split_train_validate=30)
    # t = TestSet(test)
    # t.copy_file()
    # t.set_exclude_file(['.txt', '.csv'])
    # dataset, category = d.get_dataset()
    # print(dataset)
    # print(category)
    # t.set_exclude_file('.tiff')
    #
    # valid_images = [".jpg", ".gif", ".png"]
    quit()
