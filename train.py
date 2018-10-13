#!usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
from face_recognition import FaceRecognition

if __name__ == '__main__':
    # parsing argument script
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', action='store', dest='traindataset', help='Original folder raw dataset')
    parser.add_argument('-v', '--test', action='store', dest='validataset', help='Original folder test dataset')
    parser.add_argument('-e', '--epoch', action='store', dest='epochs', type=int, help='Original folder test dataset')
    parser.add_argument('-b', '--batch', action='store', dest='batch_size', type=int, help='', default=32)
    parser.add_argument('-i', '--image size', action='store', dest='image_size', type=int, nargs=2, default=[224, 224], help='')

    args = parser.parse_args()
    # split argument in local var
    train_folder = args.traindataset
    valid_folder = args.validataset
    epochs = args.epochs
    batch = args.batch_size
    image_width = args.image_size[0]
    image_height = args.image_size[1]
    if train_folder is not None and valid_folder is not None:
        print("ok")

    else:
        print(parser.print_help())
        sys.exit()



    test = FaceRecognition(epochs=10, batch_size=32, image_width=224, image_height=224)
    test.create_img_generator()
    test.set_train_generator(train_dir=train_folder)
    test.set_valid_generator(valid_dir=valid_folder)

    # prepare the model
    test.set_face_recognition_model(pretrained_model='xception', weights='imagenet')
    name = "xception_test_{:d}"
    # train fit
    test.train_and_fit_model(name)
    del test

    #else:
    #    parser.print_help()

