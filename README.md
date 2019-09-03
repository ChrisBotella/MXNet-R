# MXNet-R

Introduction to the MXNet-R package on a supervised image classification problem from the well known [CIFAR-100 dataset to download here](https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz).

MXNet-R library enables to use, developp and learn most state of the art deep learning models for this type of task. Just check and run the script atelier_Mxnet.R.

## Package Installation

To versions of the package exist for R, CPU and GPU versions. The GPU version include the use of CPU computations, but require further dependencies, and may be harder to install depending on your system. Please check the installation instructions from the [official website](https://mxnet.incubator.apache.org/versions/master/install/)

## Pre-trained model

Running the model to reach sastifying predictive performance may be long on a personal computer. To accelerate this process, I have pre-trained the model and made it available here. One just have to download files __finish2-symbol.json__ and __finish2-0006.params__ to the directory chosen in the script and change variable __loadModel__ to __True__. 
