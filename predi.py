import os
import errno
import tensorflow as tf
from tensorflow.python.platform import gfile
from keras.optimizers import SGD
import numpy
from keras.models import load_model, model_from_json
from keras.preprocessing import image
import glob
import re
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from utilityfunction import Spinner

# suppress warning and error message tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class KerasNeuralNetwork(object):
    """
    KerasNeuralNetwork class is used to read a neural network model trained with Keras and provides several methods for
    importing a file in format: '.model', '.h5' and '.json'.
    Furthermore, before starting a new prediction, fill out the model according to the parameters used during the
    training.
    """

    def __init__(self):
        self.__spin = Spinner()
        self._model = None

    def __str__(self):
        return "KerasNeuralNetwork"

    def __compile_keras_model(self):
        """
        Before you will predict the result for a new given input you have to invoke compile method.
        After compiling, you're done to deal with new images.
        """
        self._model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])
        return self

    def set_model_from_file(self, filename, weights_file=None):
        self.__load_model_from_file(filename, weights_file)

    def __load_model_from_file(self, filename, weights_file=None):
        """
        Import trained model store as 1 file ('.model', '.h5')
        Or import the schema model in format 'json' and weights's file in format h5.
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
        self.__compile_keras_model()
        return self._model

    def predict(self, test_image):
        """
        Perform the prediction.
        :param test_image: (np-array) images in tensor form
        :return: (np-array) the probability for each class
        """
        self.__compile_keras_model()
        result = self._model.predict(test_image)
        print(result)
        return result

    def __del__(self):
        del self.__spin
        del self._model


class ModelNeuralNetwork(object):

    def __init__(self, framework, model_file_path, weight_file_path=None):
        self.framework = framework()
        self.framework.set_model_from_file(model_file_path, weight_file_path)
        self._generic_model = self.framework.get_model()


########################################################################################################################
class TensorFlowNN(object):
    def __init__(self):
        self._graph = None

    def __str__(self):
        return "TensorFlowNN"

    def __load_graph(self, model_path):
        """
        We load the protobuf file from the disk and parse it to retrieve the unserialized graph_def.
        :param model_path: (str) model's folder path
        :return: graph
        """
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

    def set_model_from_file(self, filename, weights_file=None):
        self.__load_graph(filename)

    def get_model(self):
        """
        Returns the completed keras model before start prediction
        :return: model
        """
        return self

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
    def get_input_tensor(self):
        """
        Access the input node.
        :return: input note
        """
        x = self._graph.get_tensor_by_name('input_1:0')
        return x

    def get_output_tenor(self):
        """
        Access the output node.
        :return: output node
        """
        y = self._graph.get_tensor_by_name('predictions/Softmax:0')
        return y

    def predict(self, test_image):
        #  We launch a Session
        with tf.Session(graph=self._graph) as sess:
            x = self.get_input_tensor()
            y = self.get_output_tenor()
            #     test_features = [
            #         [0.377745556, 0.009904444, 0.063231111, 0.009904444, 0.003734444, 0.002914444, 0.008633333, 0.000471111,
            #          0.009642222, 0.05406, 0.050163333, 7e-05, 0.006528889, 0.000314444, 0.00649, 0.043956667, 0.016816667,
            #          0.001644444, 0.016906667, 0.00204, 0.027342222, 0.13864]]
            #     # compute the predicted output for test_x
            pred_y = sess.run(y, feed_dict={x: test_image})
            print(pred_y)
        # return prediction
        return pred_y


########################################################################################################################
class Identification(ModelNeuralNetwork):
    def __init__(self, framework, model_file_path, weight_file_path=None):
        super(Identification, self).__init__(framework, model_file_path, weight_file_path)
        self.file_list = []
        self.img_width, self.img_height = 32, 32

    # self.__spin = Spinner()

    # def __load_tensorflow_graph(self, model_path):
    #     with tf.gfile.FastGFile(model_path, 'rb') as f:
    #         graph_def = tf.GraphDef()
    #         graph_def.ParseFromString(f.read())
    #         self.session.graph.as_default()
    #         g_in = tf.import_graph_def(graph_def)

    # def load_graph(frozen_graph_filename):
    #     # We load the protobuf file from the disk and parse it to retrieve the
    #     # unserialized graph_def
    #     with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
    #         graph_def = tf.GraphDef()
    #         graph_def.ParseFromString(f.read())
    #
    #     # Then, we can use again a convenient built-in function to import a graph_def into the
    #     # current default Graph
    #     with tf.Graph().as_default() as graph:
    #         tf.import_graph_def(
    #             graph_def,
    #             input_map=None,
    #             return_elements=None,
    #             name="prefix",
    #             op_dict=None,
    #             producer_op_list=None
    #         )
    #     return graph
    #
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
    # # We access the input and output nodes
    # x = graph.get_tensor_by_name('prefix/input_neurons:0')
    # y = graph.get_tensor_by_name('prefix/prediction_restore:0')
    #
    # # We launch a Session
    # with tf.Session(graph=graph) as sess:

    # writer = tf.summary.FileWriter('/Users/francesco/PycharmProjects/NeuralNetwork/output-checkpoint')
    # writer.add_graph(sess.graph)
    # writer.flush()
    # writer.close()

    # for op in sess.graph.get_operations():
    #   print(op)
    #    sess.close()
    # tf.Session().close()

    def _images_to_tensor(self, picture):
        """
        Given in input an image generates a tensor of the same.
        :param picture: (str) picture's path
        :return: (numpy array) codified pictures
        """
        test_image = image.load_img(picture, target_size=(self.img_width, self.img_height))
        test_image = image.img_to_array(test_image)
        test_image = numpy.expand_dims(test_image, axis=0)
        return test_image

    def load_images(self, directory_path):
        """
        This method accepts uploading images or folders to predict and test new images
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
            print("eccomi", result)
    #             result = self.model.predict(self._images_to_tensor(test_image))
    #             print(result)
    #             # self.show_image("test", test_image, result)
    #     else:
    #         # with tf.Session() as sess:
    #         self.session.run(tf.global_variables_initializer())
    #         for test_image in self.file_list:
    #             test = self._images_to_tensor(test_image)
    #             x_batch = test.reshape(1, self.img_width, self.img_height, 3)
    #
    #             y_pred = self.session.graph.get_tensor_by_name("predictions/Softmax:0")
    #
    #             # Let's feed the images to the input placeholders
    #             x = self.session.graph.get_tensor_by_name("input_1:0")
    #             # y_true = graph.get_tensor_by_name("y_true:0")
    #             # y_test_images = np.zeros((1, 2))
    #
    #             # feed_dict_testing = {x: x_batch, y_true: y_test_images}
    #             result = self.session.run(y_pred, {x: test})
    #             print(result)
    #
    # def show_image(self, name, fig, result):
    #
    #     # generate figure
    #     f = plt.figure(figsize=(12, 8))
    #     # make four subplot with gridspec
    #     gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[2, 1])
    #     # center in the grid
    #     ax1 = plt.subplot(gs[0, :])
    #     ax2 = plt.subplot(gs[1, :])
    #     # first subplot ax1 display test image passed as argument
    #     ax1.set_title("Test figure: {:s}".format(name))
    #     # read the picture and plot
    #     img = plt.imread(fig)
    #     ax1.imshow(img)
    #     # 2nd subplot ax2 display the prediction
    #     ax2.set_title("Prediction")
    #     data = {0: 'cat', 1: 'dog'}
    #     names = list(data.values())  # extract name from dict
    #     values = result[0]  # extract value from prediction
    #     rects = ax2.barh(range(len(data)), values * 100, tick_label=names)
    #     for rect in rects:
    #         height = rect.get_height()
    #         ax2.text(rect.get_x() + rect.get_height() / 2., 1.05 * height,
    #                  '%d' % int(height),
    #                  ha='center', va='bottom')
    #
    #     ax2.set_xlim(0, 100)
    #     plt.show(block=False)
    #     plt.pause(0.5)
    #     # plt.close('all')

    # def __del__(self):
    #   pass
    # self.file_list.clear()
    #
    # del self.session
    # del self.model


# pass


# #########################tf
# import numpy as np
#
# # The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
#
#
# # Credit: Josh Hemann
#
#
#
# plt.rcdefaults()
#
#
# objects = ('Python', 'C++', 'Java', 'Perl', 'Scala', 'Lisp')
# y_pos = np.arange(len(objects))
# performance = [10, 8, 6, 4, 2, 1]
#
# plt.barh(y_pos, performance, align='center', alpha=0.5)
# plt.yticks(y_pos, objects)
# plt.xlabel('Usage')
# plt.title('Programming language usage')
#
# plt.show()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# from matplotlib import animation
#
#
# fig = plt.figure()
#
# x = [1,2,3,4,5]
# y = [5,7,2,5,3]
#
# data = np.column_stack([np.linspace(0, yi, 50) for yi in y])
#
# rects = plt.bar(x, data[0], color='c')
# line, = plt.plot(x, data[0], color='r')
# plt.ylim(0, max(y))
# def animate(i):
#     for rect, yi in zip(rects, data[i]):
#         rect.set_height(yi)
#     line.set_data(x, data[i])
#     return rects, line
#
# anim = animation.FuncAnimation(fig, animate, frames=len(data), interval=40)
# plt.show()


if __name__ == '__main__':
    test = Identification(framework=KerasNeuralNetwork,
                          model_file_path="/Users/francesco/PycharmProjects/NeuralNetwork/Model/dogcat.h5")
    test.load_images("/Users/francesco/Downloads/DogAndCatDataset/test/test_images/1.jpg")
    # kerasmodel = "/Users/francesco/PycharmProjects/NeuralNetwork/Model/dogcat.h5"
    # test.load_model_from_file(kerasmodel)
    # test.load_keras_model()
    # test.load_tensorflow_graph()
    # test.show_image()
    test.predict()
    del test
    tfmodel = "/Users/francesco/PycharmProjects/NeuralNetwork/Model/dogcat.pb"
    test2 = Identification(TensorFlowNN, tfmodel)
    test2.load_images("/Users/francesco/Downloads/DogAndCatDataset/test/test_images/1.jpg")
    # test2.load_model_from_file(tfmodel)
    test2.predict()
