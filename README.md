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

The packages needed are enclosed in file “requirements.txt“, to install, type in the terminal:
```shell
pip3 install -r requirements.txt
```

NB: if you want run the script with CUDA is necessary install _"tensorflow-gpu"_ by type:
```shell
pip3 install tensorflow-gpu
```

## Convert from Keras model to NCS
Type command:
```shell
python3 keras2ncsgraph.py
```
