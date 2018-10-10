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
proceed with the installation of: VirtualBox, extension package, OS.
Once completed, we configure the virtual guest machine as shown in the following pictures.

<div style="text-align:center"><img src ="../img/fig1" /></div>
<div style="text-align:center"><img src ="../img/fig2" /></div>
<div style="text-align:center"><img src ="../img/fig3" /></div>
<div style="text-align:center"><img src ="../img/fig4" /></div>
<div style="text-align:center"><img src ="../img/fig5" /></div>
<div style="text-align:center"><img src ="../img/fig6" /></div>
<div style="text-align:center"><img src ="../img/fig7" /></div>


### Prerequisites
Project based on Python 3.5 and the following packages are needed in the enclosed file “requirements.txt“.
Then to install, type in the terminal:
```shell
pip3 install -r requirements.txt
```

NB: if you want run the script with CUDA is necessary install "tensorflow-gpu" by type:
```shell
pip3 install tensorflow-gpu
```
