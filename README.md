# Deepspark_java
Pure Java Convolutional Neural Network (CNN) package combined with Apache Spark framework

DeepSpark_java is an early version of ongoing DeepSpark project (https://github.com/deepspark/deepspark) implemented in pure java and jBlas. It provides GPU Acceleration using jCublas. (<code>gpuAccel</code> option)

DeepSpark_java also supports local training running on **single machine** and **distributed (sync & async)** training aided by Apache Spark (http://spark.apache.org/)

# USAGE INSTRUCTIONS

## **Data format**

| Class | Description                                  |
| -------------- | -------------------------------------------- |
| Tensor         | Base class for Tensor. Implemented using jBlas |
| Weight         | Class for representing Network parameters |
| Sample         | Class for representing Data container |

<code>Weight</code> and <code>Sample</code> class are implemented using <code>Tensor</code>.

To load custom dataset, users should create own data loader to be compatible with <code>Sample</code> .

We provide built-in Mnist/CIFAR/ImageNet loader (See examples on <code>src/main/java/org/acl/deepspark/utils/</code>)

## **Layer format**

| Layer | Description                                  |
| -------------- | -------------------------------------------- |
| Layer          | Base interface for layers                         |
| BaseLayer      | Abstract class implementing Layer interface   |
| ConvolutionLayer | Convolutional layer |
| PoolingLayer     | Pooling (subsampling) layer |
| FullyConnectedLayer | Normal fully connected layer |

Users should use <code>LayerConf</code> to specify layer spec (LayerType, kernel width/height, stride, padding etc.)

To add more options, check on <code>src/main/java/org/acl/deepspark/nn/conf/LayerConf</code>

## **Training**
| Layer | Description                                  |
| -------------- | -------------------------------------------- |
| NeuralNet      | Class for representing overall Network. <br/> Provides methods for initializing, training and inference |
| NeuralNetRunner | Runner of <code>NeuralNet</code> on local machine |
| DistNeuralNetRunner     | Runner of <code>NeuralNet</code> in **synchronous** distributed setting |
| DistAsyncNeuralNetRunner | Runner of <code>NeuralNet</code> in **asynchronous** distributed setting |

Users should use <code>NeuralNetConf</code> to specify training spec (lr, l2_lambda, momentum, gpuAccel etc.)

To add more options, check on <code>src/main/java/org/acl/deepspark/nn/conf/NeuralNetConf</code>

For asynchronous update, simple ParameterServer/Client class are implemented. Check on <code>src/main/java/org/acl/deepspark/nn/async</code>

## **Examples**
For actual usage code, see examples on <code>src/test/java/org/acl/deepspark/nn/driver</code>

| Type | Path                                  |
| -------------- | -------------------------------------------- |
| Single Machine | MnistTest.java / CIFARTest.java |
| Distributed (sync) | DistNeuralNetRunnerTest.java |
| Distributed (async)| AsyncMnistTest.java | AsyncCIFARTest.java |

## **Publications**
Kim, Hanjoo, Jaehong Park, Jaehee Jang, and Sungroh Yoon. "DeepSpark: Spark-Based Deep Learning Supporting Asynchronous Updates and Caffe Compatibility." arXiv preprint arXiv:1602.08191 (2016).
