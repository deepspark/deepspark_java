package org.acl.deepspark.nn.driver;

import org.acl.deepspark.data.Sample;
import org.acl.deepspark.data.Weight;
import org.acl.deepspark.nn.conf.LayerConf;
import org.acl.deepspark.nn.conf.NeuralNetConf;
import org.acl.deepspark.nn.functions.ActivatorType;
import org.acl.deepspark.nn.layers.*;
import org.acl.deepspark.utils.MnistLoader;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import java.util.Arrays;
import java.util.Date;

/**
 * Created by Jaehong on 2015-08-02.
 */
public class DistNeuralNetRunnerTest {
    public static final int minibatch = 100;
    public static final int numIteration = 1200;

    public static final double learningRate = 0.1;
    public static final double decayLambda = 0.0005;
    public static final double momentum = 0.9;
    public static final double dropOut = 0.0;

    public static void main(String[] args) throws Exception {

        SparkConf conf = new SparkConf().setAppName("DistNeuralNetRunnerTest")
                                        .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer");
        Class[] reg_classes = {Weight.class, Sample.class, NeuralNet.class, BaseLayer.class,
                ConvolutionLayer.class, FullyConnectedLayer.class, PoolingLayer.class};
        conf.registerKryoClasses(reg_classes);

        JavaSparkContext sc = new JavaSparkContext(conf);

        System.out.println("Data Loading...");
        Sample[] train_sample = MnistLoader.loadFromHDFS("data/mnist/mnist_train.txt", true);
        Sample[] test_sample = MnistLoader.loadFromHDFS("data/mnist/mnist_test.txt", true);

        JavaRDD<Sample> train_data = sc.parallelize(Arrays.asList(train_sample)).cache();

        LayerConf layer1 = new LayerConf(LayerType.CONVOLUTION);
        layer1.set("numFilters", 20);
        layer1.set("filterRow", 5);
        layer1.set("filterCol", 5);
        layer1.set("activator", ActivatorType.RECTIFIED_LINEAR);

        LayerConf layer2 = new LayerConf(LayerType.POOLING);
        layer2.set("poolRow", 2);
        layer2.set("poolCol", 2);
        layer2.set("activator", ActivatorType.NONE);

        LayerConf layer3 = new LayerConf(LayerType.CONVOLUTION);
        layer3.set("numFilters", 50);
        layer3.set("filterRow", 5);
        layer3.set("filterCol", 5);
        layer3.set("activator", ActivatorType.RECTIFIED_LINEAR);

        LayerConf layer4 = new LayerConf(LayerType.POOLING);
        layer4.set("poolRow", 2);
        layer4.set("poolCol", 2);
        layer4.set("activator", ActivatorType.NONE);

        LayerConf layer5 = new LayerConf(LayerType.FULLYCONN);
        layer5.set("numNodes", 200);
        layer5.set("activator", ActivatorType.RECTIFIED_LINEAR);

        LayerConf layer6 = new LayerConf(LayerType.FULLYCONN);
        layer6.set("numNodes", 10);
        layer6.set("activator", ActivatorType.SOFTMAX);

        NeuralNet net = new NeuralNetConf()
                .setLearningRate(learningRate)
                .setDecayLambda(decayLambda)
                .setMomentum(momentum)
                .setDropOutRate(dropOut)
                .setInputDim(new int[]{1, 28, 28})
                .setOutputDim(new int[]{10})
                .addLayer(layer1)
                .addLayer(layer2)
                .addLayer(layer3)
                .addLayer(layer4)
                .addLayer(layer5)
                .addLayer(layer6)
                .build();

        DistNeuralNetRunner driver = new DistNeuralNetRunner(net).setIterations(numIteration)
                                                                 .setMiniBatchSize(minibatch);

        System.out.println("Start Learning...");
        Date startTime = new Date();
        driver.train(sc, train_data);
        Date endTime = new Date();

        System.out.println(String.format("Accuracy: %f %%", driver.printAccuracy(test_sample)));

        long time = endTime.getTime() - startTime.getTime();
        System.out.println(String.format("Training time: %f secs", (double) time / 1000));
    }


}
