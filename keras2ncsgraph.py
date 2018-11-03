#!usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import shutil
import threading
import argparse
from utilityfunction import Spinner
import tensorflow as tf
from keras import backend as kbe
from keras.models import model_from_json, load_model
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import graph_util

# suppress warning and error message tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class KerasToNCSGraph(object):
    model_ = None  # store model
    tf_model_ = None  # tensorflow meta file graph structure
    base_dir = './ModelGraph'
    tf_model_dir = base_dir + '/tf_model/'  # folder tensorflow model
    graph_dir = base_dir + '/graph_model/'  # folder graph model

    def __init__(self):
        print("Prepare conversion form Keras model to Graph model")
        self.__delete_tmp_directory()
        if not os.path.exists(self.base_dir):
            print("make directory: '{:s}'".format(self.base_dir))
            os.makedirs(self.base_dir)
        if not os.path.exists(self.tf_model_dir):
            print("make directory: '{:s}'".format(self.tf_model_dir))
            os.makedirs(self.tf_model_dir)
        if not os.path.exists(self.graph_dir):
            print("make directory: '{:s}'".format(self.graph_dir))
            os.makedirs(self.graph_dir)

        self.output_layer_name = ""
        self.input_layer_name = ""
        self.wait = Spinner()
        self.lock = threading.Lock()

    def set_keras_model_file(self, model_file='', weights_file=None, view_summary=False):
        """
        Import trained model files.
        :param model_file (str) path to file model import 'json' or model ('.model', '.h5')
        :param weights_file (str) path to file h5 import weights model
        :param view_summary (bool) print on screen the model's summary
        """

        if os.path.exists(model_file) and weights_file is None:
            self.wait.start()
            self.lock.acquire()
            try:
                if model_file.endswith((".h5", ".model")):
                    self.model_ = load_model(model_file)
            except IOError as err:
                print("Could not read file:", model_file)
                sys.exit(err)
            self.lock.release()
            self.wait.stop()
            print("Done")
        elif os.path.exists(model_file) and weights_file is not None:
            self.wait.start()
            self.lock.acquire()
            try:
                if os.path.exists(model_file) and model_file.endswith(".json"):
                    with open(model_file, 'r') as model_input:
                        model = model_input.read()
                        self.model_ = model_from_json(model)
            except IOError as err:
                print("Could not read file: ", model_file)
                sys.exit(err)
            try:
                if os.path.exists(weights_file) and weights_file.endswith(".h5"):
                    self.model_.load_weights(weights_file)  # load weights
            except IOError as err:
                print("Could not read file: ", weights_file)
                sys.exit(err)
            self.lock.release()
            self.wait.stop()
            print("Done")
            return
        else:
            print("No files found")
            sys.exit(0)

        if view_summary:
            print('Model Summary:', self.model_.summary())  # print summary model

    def convertGraph(self, numoutputs=1, prefix='k2tfout', name='model'):
        """
        Converts an HD5F file to a .pb file for use with Tensorflow.
        :param  numoutputs (int)
        :param  prefix (str) the prefix of the output aliasing
        :param  name (str) the name of ProtocolBuffer file
        """
        print("Start conversion in Protocol Buffer")
        filename = name + '.pb'
        kbe.set_learning_phase(0)
        net_model = self.model_
        # Alias the outputs in the model - this sometimes makes them easier to access in TF
        pred = [None] * numoutputs
        pred_node_names = [None] * numoutputs
        for i in range(numoutputs):
            pred_node_names[i] = prefix + '_' + str(i)
            pred[i] = tf.identity(net_model.output[i], name=pred_node_names[i])
        print('Output nodes names are: ', pred_node_names)
        if len(pred_node_names) == 1:
            self.output_layer_name = pred_node_names[0]

        sess = kbe.get_session()

        # Write the graph in human readable
        f = 'graph_def_for_reference.pb.ascii'
        tf.train.write_graph(sess.graph.as_graph_def(), self.tf_model_dir, f, as_text=True)
        print('Saved the graph definition in ascii format at: ', os.path.join(self.tf_model_dir, f))

        # Write the graph in binary .pb file
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)
        graph_io.write_graph(constant_graph, self.tf_model_dir, filename, as_text=False)
        print('Saved the constant graph (ready for inference) at: ', os.path.join(self.tf_model_dir, filename))
        self.tf_model_ = os.path.join(self.tf_model_dir, filename)
        # compile
        self.__compile_graph_model(name_graph_model_file=name)

    def __compile_graph_model(self, name_graph_model_file='model'):
        """
        Compile graph model for Neural Compute Stick (Intel Movidius) starting from Tensorflow model.
        :param name_graph_model_file (str) assign name of model exported
        """
        graph_model = self.graph_dir
        graph_model += name_graph_model_file + '.graph'  # complete graph file model (path + name)
        # assemble command
        cmd = 'mvNCCompile {0} -s 12 -in {1} -on {2} -o {3}'.format(self.tf_model_, 'input_1',
                                                                    self.output_layer_name,
                                                                    graph_model)
        print(cmd)
        # start compile
        os.system(cmd)

    def __delete_tmp_directory(self):
        if os.path.exists(self.base_dir):
            print("removing previous temporary files")
            shutil.rmtree(self.base_dir, ignore_errors=True)

    @staticmethod
    def title():
        print("     )                                                           ")
        print("  ( /(                    )                                    ) ")
        print("  )\()) (  (      )    ( /(             (  ( (      )       ( /( ")
        print("|((_)\ ))\ )(  ( /( (  )(_))(     (  (  )\))()(  ( /( `  )  )\())")
        print("|_ ((_)((_|()\ )(_)))\((_)  )\ )  )\ )\((_))(()\ )(_))/(/( ((_)\ ")
        print("| |/ (_))  ((_|(_)_((_)_  )_(_/( ((_|(_)(()(_|(_|(_)_((_)_\| |(_)")
        print("| ' </ -_)| '_/ _` (_-</ /| ' \)) _|(_-< _` | '_/ _` | '_ \) ' \ ")
        print("|_|\_\___||_| \__,_/__/___|_||_|\__|/__|__, |_| \__,_| .__/|_||_|")
        print("                                     |___/         |_|         \n")

    def __del__(self):
        del self.model_
        del self.tf_model_


if __name__ == '__main__':
    # parsing argument script
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Convert Keras model file (.h5, json, .model) in Graph model file.')
    parser.add_argument('-k', '--keras',
                        metavar='file',
                        action='store',
                        dest='kerasmodel',
                        type=str,
                        help='requires keras model file')

    parser.add_argument('-w', '--weights',
                        metavar='file',
                        action='store',
                        dest='weights',
                        type=str,
                        required=False,
                        help='requires weights model file as: model.json and weights.h5')

    parser.add_argument('-n', '--name',
                        metavar='name',
                        action='store',
                        dest='name',
                        default='model',
                        required=False,
                        type=str,
                        help='requires name to assign output graph file')

    args = parser.parse_args()
    # split argument in local var
    model_in = args.kerasmodel
    weights_in = args.weights
    out_name = args.name
    if model_in is not None:
        # process keras model to GRAPH
        model_converter = KerasToNCSGraph()
        if weights_in is None:
            model_converter.set_keras_model_file(model_file=model_in)
        else:
            model_converter.set_keras_model_file(model_file=model_in, weights_file=weights_in)
        model_converter.convertGraph(name=out_name)
        quit()
    else:
        parser.print_help()
        sys.exit()
