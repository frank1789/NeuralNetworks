import os
import tensorflow as tf
from tensorflow.python.platform import gfile
from keras.optimizers import SGD
import numpy
from keras.models import load_model
from keras.preprocessing import image
import glob
import re
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# suppress warning and error message tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class Identification(object):
    def __init__(self):
        # self.sess = None
        self.model = None
        self.file_list = []
        self.img_width, self.img_height = 32, 32
        # with tf.Session() as sess:
        #     self.sess = sess

    # def load_tensorflow_graph(self, ):
    #
    #         with gfile.FastGFile("/Users/francesco/PycharmProjects/NeuralNetwork/Model/dogcat.pb", 'rb') as f:
    #             graph_def = tf.GraphDef()
    #             graph_def.ParseFromString(f.read())
    #             self.sess.graph.as_default()
    #             g_in = tf.import_graph_def(graph_def)

    # writer = tf.summary.FileWriter('/Users/francesco/PycharmProjects/NeuralNetwork/output-checkpoint')
    # writer.add_graph(sess.graph)
    # writer.flush()
    # writer.close()

    # for op in sess.graph.get_operations():
    #   print(op)
    #    sess.close()
    # tf.Session().close()

    def load_keras_model(self):
        self.model = load_model("/Users/francesco/PycharmProjects/NeuralNetwork/Model/dogcat.h5")
        # Before you will predict the result for a new given input you have to invoke compile method.

        # classifier.compile(loss='your_loss', optimizer='your_optimizer', metrics=['your_metrics'])
        self.model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                           loss='categorical_crossentropy', metrics=['accuracy'])

    # After compiling, you're done to deal with new images.

    def _images_to_tensor(self, picture):
        test_image = image.load_img(picture, target_size=(self.img_width, self.img_height))
        test_image = image.img_to_array(test_image)
        test_image = numpy.expand_dims(test_image, axis=0)
        return test_image

    def show_image(self, name, fig, result):

        # generate figure
        f = plt.figure(figsize=(12, 8))
        # make four subplot with gridspec
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[2, 1])
        # center in the grid
        ax1 = plt.subplot(gs[0,:])
        ax2 = plt.subplot(gs[1,:])
        # first subplot ax1 display test image passed as argument
        ax1.set_title("Test figure: {:s}".format(name))
        # read the picture and plot
        img = plt.imread(fig)
        ax1.imshow(img)
        # 2nd subplot ax2 display the prediction
        ax2.set_title("Prediction")
        data = {0: 'cat', 1: 'dog'}
        names = list(data.values())  # extract name from dict
        values = result[0]  # extract value from prediction
        rects = ax2.barh(range(len(data)), values * 100, tick_label=names)
        for rect in rects:
            height = rect.get_height()
            ax2.text(rect.get_x() + rect.get_height() / 2., 1.05 * height,
                    '%d' % int(height),
                    ha='center', va='bottom')

        ax2.set_xlim(0, 100)
        plt.show(block=False)
        plt.pause(0.5)
        #plt.close('all')

    def load_images(self, directory_path):
        picturePath = r"/Users/francesco/Downloads/DogAndCatDataset/test/test_images/20.jpg"
        # img_width, img_height = 32, 32
        test_image = image.load_img(picturePath, target_size=(self.img_width, self.img_height))
        test_image = image.img_to_array(test_image)
        test_image = numpy.expand_dims(test_image, axis=0)
        # test_image = test_image.reshape(img_width, img_height)

        for file in glob.glob(directory_path + "/*.jpg"):
            self.file_list.append(file)
            # print(file)

        self.file_list.sort()
        # print(file_list)

    def predict(self):

        if self.model is not None:
            for test_image in self.file_list:
                result = self.model.predict(self._images_to_tensor(test_image))
                print(result)
                self.show_image("test", test_image, result)



        else:
            pass
            # with tf.Session() as sess:
            # self.sess.run(tf.global_variables_initializer())
            # for test_image in self.file_list:
            #     test = self._images_to_tensor(test_image)
            #     x_batch = test.reshape(1, self.img_width, self.img_height, 3)
            #
            #     y_pred = self.sess.graph.get_tensor_by_name("predictions/Softmax:0")
            #
            #     # Let's feed the images to the input placeholders
            #     x = self.sess.graph.get_tensor_by_name("input_1:0")
            # # y_true = graph.get_tensor_by_name("y_true:0")
            # # y_test_images = np.zeros((1, 2))
            #
            # # feed_dict_testing = {x: x_batch, y_true: y_test_images}
            #     result = self.sess.run(y_pred, {x: test})
            #     print(result)


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
    test = Identification()
    test.load_images("/Users/francesco/Downloads/DogAndCatDataset/test/test_images")
    test.load_keras_model()
    # test.load_tensorflow_graph()
    # test.show_image()
    test.predict()
