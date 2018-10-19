[![License:MIT](https://img.shields.io/packagist/l/doctrine/orm.svg)](License.md)
![Python](https://img.shields.io/badge/Python-3.5-orange.svg)
# Neural Network

## Embedded System Project

Final project of the Embedded System course based on the realization of a neural network and its implementation on
[Raspberry Pi model 3](https://www.raspberrypi.org/products/raspberry-pi-3-model-b/) and [Intel Movidius](https://software.intel.com/en-us/neural-compute-stick) neural compute stick.

## Install virtual envoirments
Since the development kit provided by intel works in Linux Ubuntu environment it is recommended to install and configure a virtual machine:
+ download [VirtualBox](https://www.virtualbox.org/wiki/Downloads) and VirtualBox Extension Pack
+ download image iso Linux [Ubuntu](https://www.ubuntu-it.org/download) 16.04.4 LTS

### Installation & configuration
Proceed with the installation of: VirtualBox, extension package, OS.

Once completed, we configure the virtual guest machine as shown in the following pictures.

<div style="text-align:center"><img src ="https://github.com/frank1789/NeuralNetworks/blob/master/img/fig1.png" /></div>
<div style="text-align:center"><img src ="https://github.com/frank1789/NeuralNetworks/blob/master/img/fig2.png" /></div>
<div style="text-align:center"><img src ="https://github.com/frank1789/NeuralNetworks/blob/master/img/fig3.png" /></div>
<div style="text-align:center"><img src ="https://github.com/frank1789/NeuralNetworks/blob/master/img/fig4.png" /></div>
<div style="text-align:center"><img src ="https://github.com/frank1789/NeuralNetworks/blob/master/img/fig5.png" /></div>
<div style="text-align:center"><img src ="https://github.com/frank1789/NeuralNetworks/blob/master/img/fig6.png" /></div>
<div style="text-align:center"><img src ="https://github.com/frank1789/NeuralNetworks/blob/master/img/fig7.png" /></div>


### Prerequisites
Project based on:

+ Python 3.5.2
+ Keras 2.2.0
+ numpy
+ [tensorflow](https://www.tensorflow.org/install/) (tensorflow-gpu) 1.10.0  

The packages needed are enclosed in file “requirements.txt“, to install, type in the terminal:
```shell
pip3 install -r requirements.txt
```

NB: if you want run the script with CUDA is necessary install _"tensorflow-gpu"_ by type:
```shell
pip3 install tensorflow-gpu
```

### Install Intel Movidius sdk

To install NCSDK 2.x you can use the following command to clone the [ncsdk2](https://github.com/movidius/ncsdk/tree/ncsdk2) branch
```shell
git clone -b ncsdk2 https://github.com/movidius/ncsdk.git
```

#### Installation

The provided Makefile helps with installation. Clone the [repository](https://github.com/movidius/ncsdk/tree/ncsdk2) and then run the following command to install the NCSDK:
```shell
make install
```
#### Examples

The Neural Compute SDK also includes examples. After cloning and running 'make install,' run the following command to install the examples:
```shell
make examples
```
## Make dataset & Training Neural Networks
This script allows you to organize a dataset, downloaded from the internet or made in-house, as a structure of folders containing sets for training, validation and testing of the neural network.
<div style="text-align:center"><img src ="https://github.com/frank1789/NeuralNetworks/blob/master/img/structure.png" /></div>

This structure is congenial for use with Keras specifically with the <em>**[ flow_from_directory](https://keras.io/preprocessing/image/)**</em> method.
Be aware of the fact that if the folders are empty, the result will be a reduced dataset because it will automatically skip.
It is necessary to pass in argument:
- absolute path folder containing the raw dataset (-d);
- absolute path folder containing the raw test set (-t);
- integer value between 0 100 for dividing the dataset (-s).
```shell
python3 makeDataset.py -d ./data -t ./test -s 30
```
After this it is possible to begin to train the neural network through the script 'name' passing in argument:

| Argument |  <nobr>Long Description</nobr> | Help |
|:--------:|--------------|--------|
|-d| --dataset |requires path to train folder|
|-v| --validate|requires path to validate folder|
|-e| --epoch   |requires number of epochs, one forward pass and one backward <br>pass of all the training examples|
|-b| --batch   |requires batch size number of samples that will be <br>propagated through the network|
|-n| <nobr>--neuralnetwork</nobr>|requires to specify an existing neural network as <br>VGG, Inception, ResNet, etc|
|-f| --finetuning|requires the percentage of layers to be trained, taking weights <br>of a trained neural network and use it as initialization for a new<br> model being trained on data from the same domain|
|-i| --imagesize|requires to specify the width and height dimensions of the images|
```shell
python3 train.py -d ..data/train -v ../data/validate -e 10
```
The following neural networks are available within the script
+ VGG16 (lower case for script argument)
+ VGG19 (lower case for script argument)
+ InceptionV3 (aka *'inception'* argument script)
+ Xception (lower case for script argument)
+ ResNet50 (aka *'resnet50'* argument script)

#### [Using GPU](https://www.tensorflow.org/guide/using_gpu)
Using this script, the entire GPU memory is mapped as described below.
##### Allowing GPU memory growth
By default, TensorFlow maps nearly all of the GPU memory of all GPUs (subject to CUDA_VISIBLE_DEVICES) visible to the process. This is done to more efficiently use the relatively precious GPU memory resources on the devices by reducing memory fragmentation.

In some cases it is desirable for the process to only allocate a subset of the available memory, or to only grow the memory usage as is needed by the process. TensorFlow provides two Config options on the Session to control this.

The first is the **allow_growth** option, which attempts to allocate only as much GPU memory based on runtime allocations: it starts out allocating very little memory, and as Sessions get run and more GPU memory is needed, we extend the GPU memory region needed by the TensorFlow process. Note that we do not release memory, since that can lead to even worse memory fragmentation. To turn this option on, set the option in the ConfigProto by a [script optimized]() already implemented:
```python
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True     # to log device placement (on which device the operation ran)
config.allow_soft_placement = True      # search automatically free GPU
sess = tf.Session(config=config)
kbe.set_session(sess)                   # set this TensorFlow session as the default session for Keras
```

## Convert from Keras model to NCS
Type command:
```shell
python3 keras2ncsgraph.py
```
