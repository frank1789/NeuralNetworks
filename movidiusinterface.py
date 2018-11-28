#! /usr/bin/env python3
# -*- coding:utf-8 -*-

# Python script to open and close a single NCS device API v2
from mvnc import mvncapi
import os
import errno
import numpy as np

# suppress warning and error message tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


# main entry point for the program
class MovidiusInterface(metaclass=Singleton):
    def __init__(self):
        # set the logging level for the NC API
        mvncapi.global_set_option(mvncapi.GlobalOption.RW_LOG_LEVEL, mvncapi.LogLevel.DEBUG)
        # get a list of names for all the devices plugged into the system
        device_list = mvncapi.enumerate_devices()
        if not device_list:
            raise Exception("Error - No neural compute devices detected.")

        else:
            print(len(device_list), "neural compute devices found!")

        # Get a list of valid device identifiers
        device_list = mvncapi.enumerate_devices()
        # Create a Device instance for the first device found
        self._device = mvncapi.Device(device_list[0])
        # Open communication with the device
        # try to open the device.  this will throw an exception if someone else
        # has it open already
        try:
            self._device.open()
            print("Hello NCS! Device opened normally.")
        except Exception:
            raise Exception("Error - Could not open NCS device.")

    def __del__(self):
        # Close the device and destroy the device handle
        try:
            self._device.close()
            self._device.destroy()
            print("Goodbye NCS! Device closed normally.")
            print("NCS device working.")
        except:
            raise Exception("Error - could not close NCS device.")
        pass


class GraphNeuralNetwork(MovidiusInterface):
    def __init__(self):
        super(GraphNeuralNetwork, self).__init__()
        self.__fifo_in = None
        self.__fifo_out = None
        self.graph = None

    def __str__(self):
        return "GraphNeuralNetwork"

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
        Returns the completed keras model before start prediction
        :return model: (object)
            """
        return self

    def __load_graph(self, filename):
        """
        Load a graph file onto the NCS device
        :return:
        """
        # Read the graph file into a buffer
        if not os.path.exists(filename):
            if filename.endswith(".graph"):
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)

        with open(filename, mode='rb') as f:
            graph_buffer = f.read()
        # Load the graph buffer into the NCS
        self.graph = mvncapi.Graph(filename)
        # Set up fifos
        self.__fifo_in, self.__fifo_out = self.graph.allocate_with_fifos(self._device, graph_buffer)

    def predict(self, test_image):
        """
        Read & print inference results from the NCS
        :param test_image:
        """
        # The first inference takes an additional ~20ms due to memory
        # initializations, so we make a 'dummy forward pass'.
        self.graph.queue_inference_with_fifo_elem(self.__fifo_in, self.__fifo_out, test_image, None)
        output, userobj = self.__fifo_out.read_elem()
        # Load the image as an array
        self.graph.queue_inference_with_fifo_elem(self.__fifo_in, self.__fifo_out, test_image, None)
        # Get the results from NCS
        output, userobj = self.__fifo_out.read_elem()

        # Get execution time
        inference_time = self.graph.get_option(mvncapi.GraphOption.RO_TIME_TAKEN)
        print("Execution time: {:5.5f} ms".format(np.sum(inference_time)))
        print(output)
        return output

    def _clean(self):
        """
        Close and clean up fifos, graph
        :param self:
        """
        if self.__fifo_in is not None and self.__fifo_out is not None and self.graph is not None:
            self.__fifo_in.destroy()
            self.__fifo_out.destroy()
            self.graph.destroy()
        else:
            pass

    def __del__(self):
        self._clean()
        # Close the device and destroy the device handle
        try:
            self._device.close()
            self._device.destroy()
            print("Goodbye NCS! Device closed normally.")
            print("NCS device working.")
        except:
            raise Exception("Error - could not close NCS device.")


if __name__ == '__main__':
    a = GraphNeuralNetwork()
