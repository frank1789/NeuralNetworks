[![License:MIT](https://img.shields.io/packagist/l/doctrine/orm.svg)](License.md)
![Python](https://img.shields.io/badge/Python-3.5-orange.svg)
# Neural Network

## Embedded System Project

Final project of the Embedded System course based on the realization of a neural network and its implementation on
[Raspberry Pi model 3](https://www.raspberrypi.org/products/raspberry-pi-3-model-b/) and [Intel Movidius](https://software.intel.com/en-us/neural-compute-stick) neural compute stick.

## Install
Refer to [Readme](https://github.com/frank1789/NeuralNetworks/blob/master/README.md)

## Know issue

### Training
Cluster training (HPC) with Altair PBS does not allow saving the model file in
Keras format (*Hierarchical Data Format .h5*), so we use an internal routine to
export the model to the TensorFlow
format (*protobuff .pb*).
To guarantee that the model in keras format is compatible, it is necessary to insert the following script:
```python
kbe.set_learning_phase(0)
```
or
```python
K.set_learning_phase(0)
```
otherwise it will not be possible to convert the model into a graph format, as it will be lost in final values in the forecast.

### GPU
There may be memory allocation problems in the GPU, at the moment this solution
is used in the file *face_recognition.py*
```Python
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8, allow_growth=False)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
```
#### alterantive

Using this script, the entire GPU memory is mapped as described below.

##### Allowing GPU memory growth
By default, TensorFlow maps nearly all of the GPU memory of all GPUs (subject to
   CUDA_VISIBLE_DEVICES) visible to the process. This is done to more
   efficiently use the relatively precious GPU memory resources on the devices
   by reducing memory fragmentation.

In some cases it is desirable for the process to only allocate a subset of the
available memory, or to only grow the memory usage as is needed by the process.
TensorFlow provides two Config options on the Session to control this.

The first is the **allow_growth** option, which attempts to allocate only as
much GPU memory based on runtime allocations: it starts out allocating very
little memory, and as Sessions get run and more GPU memory is needed, we extend
the GPU memory region needed by the TensorFlow process. Note that we do not
release memory, since that can lead to even worse memory fragmentation.
To turn this option on, set the option in the ConfigProto by a script optimized
already implemented:
```python
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True      # to log device placement (on which device the operation ran)
config.allow_soft_placement = True      # search automatically free GPU
sess = tf.Session(config=config)
kbe.set_session(sess)                   # set this TensorFlow session as the default session for Keras
```

Refer documentation [Using GPU](https://www.tensorflow.org/guide/using_gpu).

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Embedded system lab @ UNITN
* HPC Cluster | ICTS - University of Trento - ICTS@unitn
