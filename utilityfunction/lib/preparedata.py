#!usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import shutil
import re
import fnmatch
import random
import argparse
from .loader import Spinner


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

    def split_in_folder_by_namefiles(self, filename, dest_dir):
        """
        Method to read the fileanme and split in folder by name
        :param filename: (str) pathfolder file
        :param dest_dir: (str) destination directory
        """
        # regex ([A-z]\w+[^_^.])(\.?\_?)([0-9])+(.png|.jpg|.gif)$
        match = re.search(r"(?P<filename>[A-z]\w+)(\.?_?)(?P<filenumber>[0-9])+(?P<extesion>.\w+)", filename)
        folderName = os.path.join(dest_dir, match.group('filename'))
        if not os.path.exists(folderName):
            os.mkdir(folderName)
            shutil.copy(os.path.join('folder', filename), folderName)
        else:
            shutil.copy(os.path.join('folder', filename), folderName)

    def make_folder_category(self, file, default_folder):
        """
        This method extract the last folder before the file in porcess
        :param  file: (str) path input file
        :return dest_dir: (str) path destination file
        """
        root_dir = os.path.dirname(file)
        path_sep = os.path.sep
        components = root_dir.split(path_sep)[-1]
        dest_dir = os.path.join(default_folder, components)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        else:
            pass
        return dest_dir

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

    def copy_file(self):
        pass

    def __del__(self):
        pass


class DataSet(PrepareDataset):
    raw_dataset = None
    __list_files = []
    __processed = False

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

        self.__spin = Spinner()
        self.__split_by_name = False

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

        if 0 < split_train_validate <= 100:
            split_train_validate /= 100
            for root, dir, f in os.walk(self.raw_dataset):
                # [f for f in os.listdir(path) if f.endswith('.txt')]
                files = [files for files in f if files not in self._exclude_ext]

                if files != []:
                    files.sort()
                    # Amount of random files you'd like to select
                    random_amount = int((len(files) * split_train_validate) + 1)
                    root_dir = root
                    path_sep = os.path.sep
                    components = root_dir.split(path_sep)[-1]
                    mess = "Total files in {:30s}: {:5d},".format(components, len(files))
                    mess += "\tsplit to {:3d}%,".format(int(split_train_validate * 100))
                    mess += "\tfiles in validate folder: {:5d}".format(random_amount)
                    print(mess)

                    for x in range(random_amount):
                        if len(files) == 0:
                            break
                        else:
                            outfile = os.path.join(root, random.choice(files))
                            if self.__split_by_name:
                                self.split_in_folder_by_namefiles(filename=outfile, dest_dir=self._default_validate)
                            else:
                                dest = self.make_folder_category(outfile, default_folder=self._default_validate)
                                shutil.copy(outfile, dest)

        else:
            print("train_split_validate must be a number to divide the training set must be 0 and 100")
            sys.exit()

    def copy_file(self, split_train_validate=30):
        """
        Copy the in correct folder splitted in train and validate
        :param split_train_validate: (int) percentul to divide set
        """
        self.__list_files = self._scan_folder(self.raw_dataset)
        count = 1
        totaldir = self.get_num_subfolders(self.raw_dataset) + 1
        # copy train folder
        for dirpath, dirnames, files in os.walk(self.raw_dataset):

            if dirnames == [] and not self.__processed:
                # check if train folder contains only files
                print('\nStart copy train folder')
                self.__spin.start()
                self.__list_files = self._scan_folder(self.raw_dataset)

                for file in files:
                    filepath = os.path.join(dirpath, file)
                    self.split_in_folder_by_namefiles(filepath, self._default_train)

                self.__spin.stop()
                print("Done")
                self.__split_by_name = True

            elif dirnames is not []:
                # check if train folder contains subfolder
                print('\nStart copy train folders {:5d} of {:5d}'.format(count, totaldir))
                self.__spin.start()

                for file in files:
                    filepath = os.path.join(dirpath, file)
                    dest_dir = self.make_folder_category(filepath, default_folder=self._default_train)
                    shutil.copy(filepath, dest_dir)

                self.__spin.stop()
                print("Done")
                self.__processed = True
                count += 1

            else:
                print("Unrecognized tree folder structure")
                sys.exit()

        # make validate folder
        self.make_validate_dir(split_train_validate)

    def __del__(self):
        self.__list_files.clear()
        del self.__spin
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

        self.__spin = Spinner()

    def copy_file(self):
        print("Start copy test folder")
        self.__spin.start()
        for file in self.__list_files:
            shutil.copy(file, self._default_test)

        self.__spin.stop()
        print("Done")

    def __del__(self):
        self.__list_files.clear()
        del self.__spin
        pass


if __name__ == '__main__':
    # parsing argument script
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', action='store', dest='rawdataset', help='Original folder raw dataset')
    parser.add_argument('-t', '--test', action='store', dest='rawtest', help='Original folder test dataset')
    parser.add_argument('-s', '--split', action='store', dest='split', default=30,
                        help='Split percentage train and validate set.')
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
