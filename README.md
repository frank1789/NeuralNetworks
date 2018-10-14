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

<div style="text-align:center"><img src ="https://github.com/frank1789/NeuralNetworks/blob/feature/readme/img/fig1.png" /></div>
<div style="text-align:center"><img src ="https://github.com/frank1789/NeuralNetworks/blob/feature/readme/img/fig2.png" /></div>
<div style="text-align:center"><img src ="https://github.com/frank1789/NeuralNetworks/blob/feature/readme/img/fig3.png" /></div>
<div style="text-align:center"><img src ="https://github.com/frank1789/NeuralNetworks/blob/feature/readme/img/fig4.png" /></div>
<div style="text-align:center"><img src ="https://github.com/frank1789/NeuralNetworks/blob/feature/readme/img/fig5.png" /></div>
<div style="text-align:center"><img src ="https://github.com/frank1789/NeuralNetworks/blob/feature/readme/img/fig6.png" /></div>
<div style="text-align:center"><img src ="https://github.com/frank1789/NeuralNetworks/blob/feature/readme/img/fig7.png" /></div>


### Prerequisites
Project based on:

+ Python 3.5.2
+ Keras 2.2.0
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
<div style="text-align:center"><img src ="https://github.com/frank1789/NeuralNetworks/blob/feature/readme/img/structure.png" /></div>

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
|:--------:|:--------------:|--------|
|
|-d| --dataset |requires path to train folder|
|-v| --validate|requires path to validate folder|
|-e| --epoch   |requires number of epochs, one forward pass and one backward pass of all the training examples|
|-b| --batch   |requires batch size number of samples that will be propagated through the network|
|-n| --neuralnetwork|requires to specify an existing neural network as VGG, Inception, ResNet, etc|
|-f| --finetuning|requires the percentage of layers to be trained, taking weights of a trained neural network and use it as initialization for a new model being trained on data from the same domain|
|-i| --imagesize|requires to specify the width and height dimensions of the images|
```shell
python3 python3 train.py -d /Users/francesco/PycharmProjects/KerasTest/dataLittle/train -v /Users/francesco/PycharmProjects/KerasTest/dataLittle/validate -e 10
```
The following neural networks are available within the script
+ VGG16 (lower case for script argument)
+ VGG19 (lower case for script argument)
+ InceptionV3 (aka *'inception'* argument script)
+ Xception (lower case for script argument)
+ ResNet50 (aka *'resnet50'* argument script)

## Convert from Keras model to NCS
Type command:
```shell
python3 keras2ncsgraph.py
```
