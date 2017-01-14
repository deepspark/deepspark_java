package org.acl.deepspark.nn.driver;

import jcuda.jcublas.JCublas;
import org.acl.deepspark.data.Sample;
import org.acl.deepspark.data.Weight;
import org.acl.deepspark.data.WeightType;
import org.acl.deepspark.nn.conf.LayerConf;
import org.acl.deepspark.nn.conf.NeuralNetConf;
import org.acl.deepspark.nn.functions.ActivatorType;
import org.acl.deepspark.nn.layers.*;
import org.acl.deepspark.utils.CIFARLoader;
import org.acl.deepspark.utils.GPUUtils;
import org.acl.deepspark.utils.MnistLoader;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import java.net.InetAddress;
import java.util.Arrays;
import java.util.Date;

/**
 * Created by Jaehong on 2015-10-05.
 */
public class AsyncMnistTest {
    public static final int minibatch = 100;
    public static final int numIteration = 6000;

    public static final double learningRate = 0.01;
    public static final double decayLambda = 0.0005;
    public static final double momentum = 0.9;
    public static final double dropOut = 0.0;
    public static final double gpuAccel = 0.0;

    public static void main(String[] args) throws Exception {
        if(gpuAccel == 1.0) {
            JCublas.cublasInit();
            GPUUtils.preAllocationMemory();
        }

        SparkConf conf = new SparkConf().setAppName("AsyncMnistTest")
                .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer");
        Class[] reg_classes = {Weight.class, Sample.class, NeuralNet.class, BaseLayer.class,
                ConvolutionLayer.class, FullyConnectedLayer.class, PoolingLayer.class};
        conf.registerKryoClasses(reg_classes);

        JavaSparkContext sc = new JavaSparkContext(conf);

        System.out.println("Data Loading...");
        Sample[] train_sample = MnistLoader.loadFromHDFS("data/mnist/mnist_train.txt", true);
        Sample[] test_sample = MnistLoader.loadFromHDFS("data/mnist/mnist_test.txt", true);

        JavaRDD<Sample> train_data = sc.parallelize(Arrays.asList(train_sample)).cache();

        LayerConf conv1 = new LayerConf(LayerType.CONVOLUTION)
                .set("num_output", 10)
                .set("kernel_row", 5)
                .set("kernel_col", 5)
                .set("stride", 1)
                .set("zeroPad", 0)
                .set("weight_type", WeightType.XAVIER)
                .set("activator", ActivatorType.RECTIFIED_LINEAR);

        LayerConf pool1 = new LayerConf(LayerType.POOLING)
                .set("kernel_row", 2)
                .set("kernel_col", 2)
                .set("stride", 2)
                .set("activator", ActivatorType.NONE);

        LayerConf conv2 = new LayerConf(LayerType.CONVOLUTION)
                .set("num_output", 20)
                .set("kernel_row", 5)
                .set("kernel_col", 5)
                .set("stride", 1)
                .set("zeroPad", 0)
                .set("weight_type", WeightType.XAVIER)
                .set("activator", ActivatorType.RECTIFIED_LINEAR);

        LayerConf pool2 = new LayerConf(LayerType.POOLING)
                .set("kernel_row", 2)
                .set("kernel_col", 2)
                .set("stride", 2)
                .set("activator", ActivatorType.NONE);

        LayerConf full1 = new LayerConf(LayerType.FULLYCONN)
                .set("num_output", 200)
                .set("weight_type", WeightType.XAVIER)
                .set("activator", ActivatorType.RECTIFIED_LINEAR);

        LayerConf full2 = new LayerConf(LayerType.FULLYCONN)
                .set("num_output", 10)
                .set("weight_type", WeightType.XAVIER)
                .set("activator", ActivatorType.SOFTMAX);

        NeuralNet net = new NeuralNetConf()
                .setLearningRate(learningRate)
                .setDecayLambda(decayLambda)
                .setMomentum(momentum)
                .setDropOutRate(dropOut)
                .setGpuAccel(gpuAccel)
                .setInputDim(new int[]{1, 1, 28, 28})
                .setOutputDim(new int[]{10})
                .addLayer(conv1)
                .addLayer(pool1)
                .addLayer(conv2)
                .addLayer(pool2)
                .addLayer(full1)
                .addLayer(full2)
                .build();

        String serverHost = InetAddress.getLocalHost().getHostAddress();
        final int[] port = new int[] {10020, 10021};
        System.out.println("ParameterServer host: " + serverHost);
        DistAsyncNeuralNetRunner driver = new DistAsyncNeuralNetRunner(net, serverHost, port)
                .setIterations(numIteration)
                .setMiniBatchSize(minibatch);

        System.out.println("Start Learning...");
        Date startTime = new Date();
        driver.train(train_data);
        Date endTime = new Date();

        System.out.println(String.format("Accuracy: %f %%", driver.printAccuracy(test_sample)));

        if(gpuAccel == 1.0) {
            GPUUtils.clearGPUMem();
            JCublas.cublasShutdown();
        }

        long time = endTime.getTime() - startTime.getTime();
        System.out.println(String.format("Training time: %f secs", (double) time / 1000));
    }
}
