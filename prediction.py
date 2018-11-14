import os
import errno
import json
import glob
import numpy as np
from utilityfunction import Spinner
import tensorflow as tf
from keras.optimizers import SGD
from keras.models import load_model, model_from_json
from keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
import sys

# suppress warning and error message tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class KerasNeuralNetwork(object):
    """
    KerasNeuralNetwork class is used to read a neural network model trained with
    Keras and provides several methods for importing a file in format:
     - '.model';
     - '.h5';
     - '.json'.
    Furthermore, before starting a new prediction, fill out the model according
    to the parameters used during the training.
    """

    def __init__(self):
        self.__spin = Spinner()
        self._model = None
        self._config = None

    def __str__(self):
        return "KerasNeuralNetwork"

    def set_model_from_file(self, filename, weights_file=None, config_compiler=None):
        """
        Read from file the correct model.
        :param filename: (str) model file model path
        :param weights_file: (str) weight file path - optional
        :param config_compiler: (str) configuration from training - optional
        """
        self.__load_model_from_file(filename, weights_file)
        self._config = config_compiler

    def __compile_keras_model(self):
        """
        Before you will predict the result for a new given input you have to
        invoke compile method.
        After compiling, you're done to deal with new images.
        _config -> tuple
        _config[0] = compiler name
        _config[1] = learning rate
        _config[2] = momentum
        _config[3] = loss category
        _config[4] = metrics
        """
        if self._config[0] == 'SGD':
            self._model.compile(optimizer=SGD(lr=self._config[1], momentum=self._config[2]),
                                loss=self._config[3],
                                metrics=self._config[4])
        else:
            self._model.compile(optimizer=self._config[0], loss=self._config[3], metrics=self._config[4])

        return self

    def __load_model_from_file(self, filename, weights_file=None):
        """
        Import trained model store as 1 file ('.model', '.h5')
        Or import the schema model in format 'json' and weights's file in
        format h5.
        :param filename: (str) pass path model file
        :param weights_file: (str) pass path weights file
        """
        if os.path.exists(filename) and weights_file is None:
            print("Loading model, please wait")
            self.__spin.start()

            # load entire model
            if filename.endswith(('.model', '.h5')):
                self._model = load_model(filename)
                self.__spin.stop()
                print("Done")

            else:
                self.__spin.stop()
                raise ValueError("Invalid extension, supported extensions are: '.h5', '.model'")

        elif os.path.exists(filename) and weights_file is not None:
            if filename.endswith('.json') and weights_file.endswith('.h5'):
                print("Loading model, please wait")
                self.__spin.start()
                # Model reconstruction from JSON file
                with open(filename, 'r') as f:
                    self._model = model_from_json(f.read())
                # Load weights into the new model
                self._model.load_weights(weights_file)
                self.__spin.stop()
                print("Done")

            else:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), (filename, weights_file))

        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)

    def get_model(self):
        """
        Returns the completed keras model before start prediction
        :return: model
        """
        return self

    def predict(self, test_image):
        """
        Perform the prediction.
        :param test_image: (np-array) images in tensor form
        :return: (np-array) the probability for each class
        """
        self.__compile_keras_model()
        result = self._model.predict(test_image)
        print(result)
        return result[0]

    def _clean(self):
        pass

    def __del__(self):
        del self.__spin
        del self._model


class TensorFlowNeuralNetwork(object):
    """
    TensorFlowNeuralNetwork class is used to read a neural network model trained
    with TensorFlow and provides several methods for importing a file
    in format: '.pb'.
    Furthermore, before starting a new prediction, fill out the model according
    to the parameters used during the training.
    """

    def __init__(self):
        self._graph = None

    def __str__(self):
        return "TensorFlowNeuralNetwork"

    def __load_graph(self, model_path):
        """
        We load the protobuf file from the disk and parse it to retrieve the
        unserialized graph_def.
        :param model_path: (str) model's folder path
        :return: graph
        """
        print("Read model, please wait...")
        with tf.gfile.GFile(model_path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # Then, we can use again a convenient built-in function to import a graph_def into the
        # current default Graph
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(
                graph_def,
                input_map=None,
                return_elements=None,
                name="prefix",
                # op_dict=None,
                producer_op_list=None
            )
        self._graph = graph
        print("Done")

    def set_model_from_file(self, filename, weights_file=None, config_compiler=None):
        """
        Read from file the correct model.
        :param filename: (str) model file model path
        :param weights_file: (str) weight file path - optional
        :param config_compiler: (str) configuration from training - optional
        """
        self.__load_graph(filename)

    def get_model(self):
        """
        Returns the completed tensorflow model before start prediction
        :return: model
        """
        return self

    def get_input_tensor(self):
        """
        Access the input node.
        :return: input note
        """
        # # We use our "load_graph" function
        # graph = load_graph("./models/frozen_model.pb")
        #
        # # We can verify that we can access the list of operations in the graph
        # for op in graph.get_operations():
        #     print(op.name)  # <--- printing the operations snapshot below
        #     # prefix/Placeholder/inputs_placeholder
        #     # ...
        #     # prefix/Accuracy/predictions
        #
        x = self._graph.get_tensor_by_name('prefix/input_1:0')
        return x

    def get_output_tenor(self):
        """
        Access the output node.
        :return: output node
        """
        # # We use our "load_graph" function
        # graph = load_graph("./models/frozen_model.pb")
        #
        # # We can verify that we can access the list of operations in the graph
        # for op in graph.get_operations():
        #     print(op.name)  # <--- printing the operations snapshot below
        #     # prefix/Placeholder/inputs_placeholder
        #     # ...
        #     # prefix/Accuracy/predictions
        #
        y = self._graph.get_tensor_by_name('prefix/predictions/Softmax:0')
        return y

    def predict(self, test_image):
        """
        Perfom the prediction based on graph TensorFlow
        :param test_image: image in tensor fromat
        :return: (np-arry) prediction probability
        """
        with tf.Session(graph=self._graph) as sess:  # launch a Session
            x = self.get_input_tensor()
            y = self.get_output_tenor()
            # compute the predicted output for test_x
            pred_y = sess.run(y, feed_dict={x: test_image})
            print(pred_y)
        # return prediction
        return pred_y[0]

    def _clean(self):
        pass


class ModelNeuralNetwork(object):
    """
    Design Pattern Class to instantiate the correct class to decode previously
    trained models currently supports:
    - Keras ('.h5', '.model', '.json')
    - TensorFlow ('.pb', protobuff)
    - Intel Movidius ('.graph)
    """

    def __init__(self, framework, config_file_path, model_file_path, weight_file_path=None):
        # import configuration file
        with open(config_file_path, "r") as read_file:
            data = json.load(read_file)

        # init param
        self._img_width = data["image_width"]
        self._img_height = data["image_height"]
        self._label_map = data["label_map"]
        config_compiler = (data["optimizer"], data["learning_rate"], data["momentum"], data["loss"], data["metrics"])

        # init framework
        self.framework = framework()
        self.framework.set_model_from_file(model_file_path, weight_file_path, config_compiler)
        self._generic_model = self.framework.get_model()
        self.result = []
        if not os.path.exists(config_file_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), config_file_path)


class Identification(ModelNeuralNetwork):
    def __init__(self, framework, config_file_path, model_file_path, weight_file_path=None):
        self.file_list = []
        super(Identification, self).__init__(framework, config_file_path, model_file_path, weight_file_path)


    def _images_to_tensor(self, picture):
        """
        Given in input an image generates a tensor of the same.
        :param picture: (str) picture's path
        :return: (numpy array) codified pictures
        """
        test_image = image.load_img(picture, target_size=(self._img_width, self._img_height))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        return test_image

    def load_images(self, directory_path):
        """
        This method accepts uploading images or folders to predict and test new
        images.
        :param directory_path: (str) picture's folder/path
        :return:
        """
        if os.path.isdir(directory_path):
            for file in glob.glob(directory_path + "/*.jpg"):
                self.file_list.append(file)
            self.file_list.sort()

        elif os.path.isfile(directory_path):
            self.file_list.append(directory_path)
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), directory_path)

    def predict(self):

        for test_image in self.file_list:
            result = self._generic_model.predict(self._images_to_tensor(test_image))
            self.show_image(os.path.basename(test_image),
                            test_image,
                            result)

    def show_image(self, name, fig, result):

        # generate figure
        f = plt.figure(figsize=(12, 8))
        # make four subplot with gridspec
        if len(self._label_map) > 2:
            gs = gridspec.GridSpec(4, 2,)
        else:
            gs = gridspec.GridSpec(2, 2, )
        # center in the grid
        ax1 = plt.subplot(gs[0, :])
        ax2 = plt.subplot(gs[1:, :])
        # first subplot ax1 display test image passed as argument
        ax1.set_title("Test figure: {:s}".format(name))
        # read the picture and plot
        img = plt.imread(fig)
        ax1.imshow(img)
        # 2nd subplot ax2 display the prediction
        ax2.set_title("Prediction")
        data = self._label_map
        names = list(data.values())  # extract name from dict
        values = result  # extract value from prediction
        ax2.barh(range(len(data)), values * 100, tick_label=names)
        ax2.set_xlim(0, 100)
        plt.show(block=False)
        plt.pause(5)
        plt.close('all')

    def __del__(self):
        self.framework._clean()
        self.file_list.clear()



class MyArgumentParser(object):

    @staticmethod
    def title():
        print(".------..------..------..------..------..------..------.")
        print("|P.--. ||R.--. ||E.--. ||D.--. ||I.--. ||C.--. ||T.--. |")
        print("| :/\: || :(): || (\/) || :/\: || (\/) || :/\: || :/\: |")
        print("| (__) || ()() || :\/: || (__) || :\/: || :\/: || (__) |")
        print("| '--'P|| '--'R|| '--'E|| '--'D|| '--'I|| '--'C|| '--'T|")
        print("'------'`------'`------'`------'`------'`------'`------'")

    def __init__(self):
        self.title()
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                              description='You can make predictions using a trained neural'
                                                          ' network for deep learning.')

        self.parser.add_argument('--configfile',
                                 action='store',
                                 dest='configfile',
                                 type=str,
                                 help='requires path of configuration file generated during training.')
        self.parser.add_argument('--model',
                                 action='store',
                                 dest='modelfile',
                                 type=str,
                                 help='requires path of model file')
        self.parser.add_argument('--weights',
                                 action='store',
                                 dest='weights',
                                 type=str,
                                 required=False,
                                 help='requires path of weights model file')
        self.parser.add_argument('--test',
                                 action='store',
                                 dest='testfolder',
                                 type=str,
                                 required=False,
                                 help='requires path test folder')
        self.args = {}
        self.__check_input_args()

    def __check_input_args(self):
        optional = self.parser.parse_args()
        # check config file
        if optional.configfile is not None:
            self.args['configfile'] = optional.configfile

        else:
            self.parser.print_help()
            sys.exit()

        # check presence model file
        if optional.modelfile is not None:
            self.args['model'] = optional.modelfile

        else:
            self.parser.print_help()
            sys.exit()

        # check presence test folder/file
        if optional.testfolder is not None:
            self.args['test'] = optional.testfolder

        else:
            self.parser.print_help()
            sys.exit()

        self.args['weights'] = optional.weights

    def get_arguments(self):
        """
        Return parsed argument
        :return: (dict) with input user
        """
        return self.args

    def __del__(self):
        self.args.clear()


if __name__ == '__main__':
    # parsing argument
    parse = MyArgumentParser()
    parsed = parse.get_arguments()
    test = None
    # check type of model
    filename, file_extension = os.path.splitext(parsed['model'])
    if file_extension in ['.h5', '.model', '.json']:
        if parsed is None:
            test = Identification(framework=KerasNeuralNetwork,
                                  config_file_path=parsed['configfile'],
                                  model_file_path=parsed['model'])
        else:
            test = Identification(framework=KerasNeuralNetwork,
                                  config_file_path=parsed['configfile'],
                                  model_file_path=parsed['model'],
                                  weight_file_path=parsed['weights'])

    elif file_extension in ['.pb']:
        test = Identification(framework=TensorFlowNeuralNetwork,
                              config_file_path=parsed['configfile'],
                              model_file_path=parsed['model'])

    elif file_extension in ['.graph']:
        from movidiusinterface import GraphNeuralNetwork
        test = Identification(framework=GraphNeuralNetwork,
                              config_file_path=parsed['configfile'],
                              model_file_path=parsed['model'])

    else:
        print("Format not supported.")
        sys.exit()

    test.load_images(parsed['test'])
    test.predict()
    del test
    quit()
