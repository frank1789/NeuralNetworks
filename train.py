#!usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
from face_recognition import FaceRecognition


class MyArgumentParser(object):

    @staticmethod
    def title():
        print("(`-')         (`-')   (`-')  _    _      <-. (`-')_  ")
        print("( OO).->   <-.(OO )   (OO ).-/   (_)        \( OO) ) ")
        print("/    '._   ,------,)  / ,---.    ,-(`-') ,--./ ,--/  ")
        print("|'--...__) |   /`. '  | \ /`.\   | ( OO) |   \ |  |  ")
        print("`--.  .--' |  |_.' |  '-'|_.' |  |  |  ) |  . '|  |) ")
        print("   |  |    |  .   .' (|  .-.  | (|  |_/  |  |\    |  ")
        print("   |  |    |  |\  \   |  | |  |  |  |'-> |  | \   |  ")
        print("   `--'    `--' '--'  `--' `--'  `--'    `--'  `--'  ")
        print(" <-. (`-')_  (`-')  _               (`-')   (`-')")
        print("    \( OO) ) ( OO).-/      .->   <-.(OO )   (OO ).-/     <-.")
        print(" ,--./ ,--/ (,------. ,--.(,--.  ,------,)  / ,---.    ,--. )")
        print(" |   \ |  |  |  .---' |  | |(`-')|   /`. '  | \ /`.\   |  (`-')")
        print(" |  . '|  |)(|  '--.  |  | |(OO )|  |_.' |  '-'|_.' |  |  |OO )")
        print(" |  |\    |  |  .--'  |  | | |  \|  .   .' (|  .-.  | (|  '__ |")
        print(" |  | \   |  |  `---. \  '-'(_ .'|  |\  \   |  | |  |  |     |'")
        print(" `--'  `--'  `------'  `-----'   `--' '--'  `--' `--'  `-----' ")
        print(" <-. (`-')_  (`-')  _ (`-')           .->                 (`-')  <-.(`-')  ")
        print("    \( OO) ) ( OO).-/ ( OO).->    (`(`-')/`)     .->   <-.(OO )   __( OO)  ")
        print(" ,--./ ,--/ (,------. /    '._   ,-`( OO).',(`-')----. ,------,) '-'. ,--. ")
        print(" |   \ |  |  |  .---' |'--...__) |  |\  |  |( OO).-.  '|   /`. ' |  .'   / ")
        print(" |  . '|  |)(|  '--.  `--.  .--' |  | '.|  |( _) | |  ||  |_.' | |      /) ")
        print(" |  |\    |  |  .--'     |  |    |  |.'.|  | \|  |)|  ||  .   .' |  .   '  ")
        print(" |  | \   |  |  `---.    |  |    |   ,'.   |  '  '-'  '|  |\  \  |  |\   \ ")
        print(" `--'  `--'  `------'    `--'    `--'   '--'   `-----' `--' '--' `--' '--' ")

    def __init__(self):
        self.title()
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                              description='Train a neural network.')

        self.parser.add_argument('-d', '--dataset',
                                 metavar='../data/train',
                                 action='store',
                                 dest='traindataset',
                                 help='requires path to train folder')
        self.parser.add_argument('-v', '--validate',
                                 metavar='../data/validate',
                                 action='store',
                                 dest='validataset',
                                 help='requires path to validate folder')

        self.parser.add_argument('-e', '--epoch',
                                 metavar='NN',
                                 action='store',
                                 dest='epochs',
                                 type=int,
                                 help='requires number of epochs, one forward pass and one backward pass of all the '
                                      'training examples')

        self.parser.add_argument('-b', '--batch',
                                 metavar='NN',
                                 action='store',
                                 dest='batch_size',
                                 type=int,
                                 default=32,
                                 help='requires batch size number of samples that will be propagated through '
                                      'the network')

        self.parser.add_argument('-n', '-neuralnetwork',
                                 metavar='name neural network',
                                 action='store',
                                 dest='neuralnetwork',
                                 type=str,
                                 required=False,
                                 help='requires to specify an existing neural network as VGG, Inception, ResNet, etc')

        self.parser.add_argument('-f', '--finetuning',
                                 metavar='NN',
                                 action='store',
                                 dest='fine_tuning',
                                 type=int,
                                 required=False,
                                 help='requires the percentage of layers to be trained, taking weights of a trained '
                                      'neural network and use it as initialization for a new model being trained on '
                                      'data from the same domain')

        self.parser.add_argument('-i', '--imagesize',
                                 metavar='NNN',
                                 action='store',
                                 dest='image_size',
                                 type=int,
                                 nargs=2,
                                 default=[224, 224],
                                 required=False,
                                 help='requires to specify the width and height dimensions of the images')

        self.args = {}
        self.__check_input_args()

    def __check_input_args(self):
        optinal = self.parser.parse_args()
        if optinal.traindataset is not None:
            self.args['train'] = optinal.traindataset
        else:
            self.parser.print_help()
            sys.exit()

        if optinal.validataset is not None:
            self.args['valid'] = optinal.validataset
        else:
            self.parser.print_help()
            sys.exit()

        if optinal.epochs is not None:
            self.args['epochs'] = optinal.epochs
        else:
            self.parser.print_help()
            sys.exit()

        if optinal.neuralnetwork is not None:
            self.args['neuralnetwork'] = optinal.neuralnetwork
        else:
            self.args['neuralnetwork'] = ''

        if optinal.fine_tuning is not None:
            self.args['fine_tuning'] = (True, (optinal.fine_tuning / 100))
        else:
            self.args['fine_tuning'] = False

        self.args['batch'] = optinal.batch_size
        self.args['image_width'] = optinal.image_size[0]
        self.args['image_height'] = optinal.image_size[1]

    def get_arguments(self):
        out_dict = self.args
        return out_dict

    def __del__(self):
        del self.args


if __name__ == '__main__':
    # parsing arguments
    args = MyArgumentParser().get_arguments()
    neuralnetwork = FaceRecognition(epochs=args['epochs'],
                                    batch_size=args['batch'],
                                    image_width=args['image_width'],
                                    image_height=args['image_height'])
    neuralnetwork.create_img_generator()
    neuralnetwork.set_train_generator(train_dir=args['train'])
    neuralnetwork.set_valid_generator(valid_dir=args['valid'])

    # prepare the model
    if args['fine_tuning']:
        neuralnetwork.set_face_recognition_model(pretrained_model=args['neuralnetwork'],
                                                 weights='imagenet',
                                                 trainable_parameters=args['fine_tuning'][0],
                                                 num_trainable_parameters=args['fine_tuning'][1])
    else:
        neuralnetwork.set_face_recognition_model(pretrained_model=args['neuralnetwork'],
                                                 weights='imagenet')
    # name for plot
    name = args['neuralnetwork']
    # train fit
    neuralnetwork.train_and_fit_model(name)
    # clear
    del neuralnetwork
