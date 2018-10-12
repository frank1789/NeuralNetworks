#!usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import shutil
import re
import fnmatch
import random
import argparse


class PrepareDataset(object):
    """
    It is a class that allows to prepare the analysis of the dataset once imported from a source.
    Inside there are methods for manipulating the data-set.
                                           _
                                          | \_____
                                          | data  |
                                          |_______|
                                            |
                            +---------------+-------------------+       
                            |               |                   |
                           _              _                    _
                          | \_____       | \_________         | \_____
                          | train |      | validate  |        | test  |
                          |_______|      |___________|        |_______|
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

    def copy_file(self):
        pass


class DataSet(PrepareDataset):
    raw_dataset = None

    def __init__(self, raw_dataset):
        """
        Default constructor of data-set. Once invoked it checks the validity of the examined folder, proceeds in the
        generation of the list of the files of the dataset and of the relative classification.
        :param raw_dataset: (str) path's data-set.
        """
        super(DataSet, self).__init__()
        # check if folder exist, then build the database
        try:
            if os.path.exists(raw_dataset):
                self.raw_dataset = raw_dataset
        except OSError:
            if not os.path.isdir(raw_dataset):
                raise

    def get_dataset(self):
        """
        Return the data-set and sub-category.
        :returns (list) files
        :return (dict) category's data-set
        """
        return self.__list_files, self.__list_category

    def make_validate_dir(self, split_train_validate=30):
        """
        Copy from original folder and split in two folder train and validate
                            data
                             |
                    +--------+--------+
                    |                 |
                  train            validate
        :param split_train_validate: (int) indicates how much to divide the training set
        """
        try:
            if 0 < split_train_validate <= 100:
                split_train_validate /= 100
                for root, dir, files in os.walk(self.raw_dataset):
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

    def copy_file(self, split_train_validate=30):
        """
        Copy the in correct folder splitted in train and validate
        :param split_train_validate: (int) percentul to divide set
        """
        # copy train folder
        if os.path.exists(self.raw_dataset):
            self.__list_files = self._scan_folder(self.raw_dataset)
        print("Start copy train folder, please wait...")
        for file in self.__list_files:
            shutil.copy(file, self._default_train)
        print("Done")
        # make validate folder
        self.make_validate_dir(split_train_validate)

    def __del__(self):
        pass


class TestSet(PrepareDataset):
    __list_files = []  # list of files in test-set

    def __init__(self, raw_test_set):
        """
        Default constructor of data-set. Once invoked it checks the validity of the examined folder, proceeds in the
        generation of the list of the files to validate the model.
        :param raw_test_set: (str) path's test-set.
        """
        super(TestSet, self).__init__()
        if os.path.exists(raw_test_set):
            self.__list_files = self._scan_folder(raw_test_set)

    def copy_file(self):
        print("Start copy test folder, please wait...")
        for file in self.__list_files:
            shutil.copy(file, self._default_test)
        print("Done")

    def __del__(self):
        self.__list_files.clear()
        pass


if __name__ == '__main__':
    # parsing argument script
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', action='store', dest='rawdataset', help='Original folder raw dataset')
    parser.add_argument('-t', '--test', action='store', dest='rawtest', help='Original folder test dataset')
    parser.add_argument('-s', '--split', action='store', dest='split', default=30,
                        help='Split percentage train and validate set.')

    # # TODO remove
    #
    # datest1 = '/Users/francesco/Downloads/the-simpsons-characters-dataset/simpsons_dataset/'
    # test = '/Users/francesco/Downloads/the-simpsons-characters-dataset/kaggle_simpson_testset'
    #
    # d = DataSet(datest1)
    # d.copy_file(split_train_validate=30)
    # t = TestSet(test)
    # t.copy_file()
    #
    # del t
    # del d
    #
    # datates2 = r'/Users/francesco/Downloads/DogAndCatDataset/train'
    # tets2 = r'/Users/francesco/Downloads/DogAndCatDataset/test'
    #
    # d2 = DataSet(datates2)
    # d2.copy_file()
    # t2 =  TestSet(tets2)
    # t2.copy_file()
    #
    #
    # del t2
    # del d2

    ####################################################################################
    try:
        args = parser.parse_args()
        # split argument in local var
        dataset_in = args.rawdataset
        test_in = args.rawtest
        split_in = args.split
        # process dataset
        out_dataset = DataSet(dataset_in)
        out_test = TestSet(test_in)
        out_dataset.copy_file(split_train_validate=split_in)
        out_test.copy_file()
        # clean all
        del out_dataset, out_test
    except:
        parser.print_help()
        sys.exit()
