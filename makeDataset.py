#!usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
from utilityfunction import DataSet, TestSet

if __name__ == '__main__':
    # parsing argument script
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                    description='Utility to organize dataset folder')
    parser.add_argument('-d', '--dataset',
                        action='store',
                        dest='rawdataset',
                        help='Original folder raw dataset')
    parser.add_argument('-t', '--test',
                        action='store',
                        dest='rawtest',
                        help='Original folder test dataset')
    parser.add_argument('-s', '--split',
                        action='store',
                        dest='split',
                        type=int,
                        default=30,
                        help='Split percentage train and validate set.')

    args = parser.parse_args()

    # split argument in local var
    dataset_in = args.rawdataset
    test_in = args.rawtest
    split_in = args.split
    if dataset_in is not None and test_in is not None:
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
