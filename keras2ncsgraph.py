import os
import sys
import tensorflow as tf
from keras import backend as kbe
from keras.models import model_from_json, load_model

# suppress warning and error message tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class KerasModel2NCSGraph(object):
    model_ = None  # store model
    tf_model_ = None  # tensorflow meta file graph structure
    base_dir = './ModelGraph'
    tf_model_dir = './ModelGraph/tf_model/'  # folder tensorflow model
    graph_dir = './ModelGraph/graph_model'  # folder graph model

    def __init__(self):
        print("Prepare conversion form Keras model to Graph model")
        if not os.path.exists(self.base_dir):
            print("make directory: '{:s}'".format(self.base_dir))
            os.makedirs(self.base_dir)
        if not os.path.exists(self.tf_model_dir):
            print("make directory: '{:s}'".format(self.tf_model_dir))
            os.makedirs(self.tf_model_dir)
        if not os.path.exists(self.tf_model_dir):
            print("make directory: '{:s}'".format(self.graph_dir))
            os.makedirs(self.graph_dir)

    def set_keras_model_file(self, model_file):
        """Import json file and weights' file.
        :param model_file (str) path to file model
        """
        self.model_ = load_model(model_file)
        # get summary model
        print('Model Summary:', self.model_.summary())

    def set_keras_json_file(self, model_file, weights_file):
        """Import json file and weights' file.
        :param model_file (str) path to file json
        :param weights_file (str) path to file h5
        """
        with open(model_file, 'r') as model_input:
            model = model_input.read()
            self.model_ = model_from_json(model)

        # load weights:
        self.model_.load_weights(weights_file)
        # get summary model
        print('Model Summary:', self.model_.summary())

    def keras_to_tensorflow(self):
        """Convert Keras model to Tensorflow model."""
        tf_model_file = tf.train.Saver()
        with kbe.get_session() as tf_session:
            kbe.set_learning_phase(0)
            tf_model_file.save(tf_session, self.tf_model_dir + 'tf_model')

        for root, dir, files in os.walk(self.tf_model_dir):
            namefile = [file for file in files if file.endswith(".meta")][0]  # store meta file
            self.tf_model_ = os.path.join(root, namefile)

    def compile_graph_model(self, name_graph_model_file='model'):
        """Compile graph model for Neural Compute Stick (Intel Movidius) starting from Tensorflow model."""

        graph_model = self.graph_dir + '/' + name_graph_model_file + '.graph'  # complete graph file model (path + name)
        cmd = 'mvNCCompile {0}.meta -in {1} -on {2} -o {3}'.format(self.tf_model_, '0', '0', graph_model)
        print(cmd)
        # start compile
        # os.system('mvNCCompile {0}.meta -in {1} -on {2} -o {3}')
        pass
    #
    # def tf_to_graph(self, tf_model_path, model_in, model_out, graph_path):
    #     os.system('mvNCCompile {0}.meta -in {1} -on {2} -o {3}'.format(tf_model_path, model_in, model_out, graph_path))
    #     return True
    #
    # def keras_to_graph(self, model_path, model_in, model_out, weights_path, graph_path, take_tf_files = False):
    # # Getting Keras Model:
    #     keras_model = get_keras_model(model_path, weights_path)
    #
    #  # Saving TensorFlow Model from Keras Model:
    #     tf_model_path = './TF_Model/tf_model'
    #     keras_to_tf(tf_model_path)
    #
    #     tf_to_graph(tf_model_path, model_in, model_out, graph_path)
    #
    #     if take_tf_files == False:
    #         os.system('rm -rf ./TF_Model')


if __name__ == '__main__':
    # try:
    #     model_path = sys.argv[1]
    #     model_in = sys.argv[2]
    #     model_out = sys.argv[3]
    #     weights_path = sys.argv[4]
    #     graph_path = sys.argv[5]
    # except:
    #     print('Run with arguments !\nArguments:\nmodel_path model_in model_out weights_path graph_path')
    #     exit()
    #
    # keras_to_graph(model_path, model_in, model_out, weights_path, graph_path, False)
    t = KerasModel2NCSGraph()
    t.set_keras_model_file('inceptionv3-transfer-learning.model')
    t.keras_to_tensorflow()
    t.compile_graph_model()
