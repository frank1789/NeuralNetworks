#!usr/bin/env python3
# -*- coding: utf-8 -*-

# import os
# import shutil
# import random
#
# # You can get all the files in a directory like this:
# dir = r'/Users/francesco/Desktop/data/train/cat'
# outputdir = r'/Users/francesco/Desktop/data/validate/cat'
# if not os.path.exists(outputdir):
#     os.makedirs(outputdir)
#
# files = [file for file in os.listdir(dir) if os.path.isfile(os.path.join(dir, file))]
# files.sort()
# # count file in dir and select 30%
# print("total files: {:7d}, then select 30% {:7d}".format(len(files), int((len(files) * 0.3) + 1)))
#
# # Amount of random files you'd like to select
# random_amount = int((len(files) * 0.3) + 1)
# for x in range(random_amount):
#     if len(files) == 0:
#         break
#     else:
#         file = random.choice(files)
#         shutil.copy(os.path.join(dir, file), outputdir)
#
#
# # You can get all the files in a directory like this:
# dir = r'/Users/francesco/Desktop/data/train/dog'
# outputdir = r'/Users/francesco/Desktop/data/validate/dog'
# if not os.path.exists(outputdir):
#     os.makedirs(outputdir)
#
# files = [file for file in os.listdir(dir) if os.path.isfile(os.path.join(dir, file))]
# files.sort()
# # count file in dir and select 30%
# print("total files: {:7d}, then select 30% {:7d}".format(len(files), int((len(files) * 0.3) + 1)))
#
# # Amount of random files you'd like to select
# random_amount = int((len(files) * 0.3) + 1)
# for x in range(random_amount):
#     if len(files) == 0:
#         break
#     else:
#         file = random.choice(files)
#         shutil.copy(os.path.join(dir, file), outputdir)
import argparse
import sys
from utilityfunction import DataSet, TestSet

if __name__ == '__main__':
    # parsing argument script
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', action='store', dest='rawdataset', help='Original folder raw dataset')
    parser.add_argument('-t', '--test', action='store', dest='rawtest', help='Original folder test dataset')
    parser.add_argument('-s', '--split', action='store', dest='split', default=30,
                        help='Split percentage train and validate set.')

    args = parser.parse_args()

    # split argument in local var
    dataset_in = args.rawdataset
    test_in = args.rawtest
    split_in = args.split
    if dataset_in is None and test_in is not None:
        # process dataset
        out_dataset = DataSet(dataset_in)
        out_test = TestSet(test_in)
        out_dataset.copy_file(split_train_validate=split_in)
        out_test.copy_file()
        # clean all
        del out_dataset, out_test
        quit()
    else:
        parser.print_help()
        sys.exit()
