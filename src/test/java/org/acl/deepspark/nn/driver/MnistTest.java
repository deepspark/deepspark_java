package org.acl.deepspark.nn.driver;

import org.acl.deepspark.data.Sample;
import org.acl.deepspark.data.WeightType;
import org.acl.deepspark.nn.conf.LayerConf;
import org.acl.deepspark.nn.conf.NeuralNetConf;
import org.acl.deepspark.nn.functions.ActivatorType;
import org.acl.deepspark.nn.layers.LayerType;
import org.acl.deepspark.utils.MnistLoader;

import java.util.Date;

/**
 * Created by Jaehong on 2015-10-01.
 */
public class MnistTest {
    public static final int minibatch = 100;
    public static final int numIteration = 1200;

    public static final double learningRate = 0.1;
    public static final double decayLambda = 0.0005;
    public static final double momentum = 0.9;
    public static final double dropOut = 0.0;

    public static void main(String[] args) throws Exception {

        Sample[] training_data = MnistLoader.loadIntoSamples("C:/Users/Jaehong/Downloads/mnist_train.txt", true);
        Sample[] test_data = MnistLoader.loadIntoSamples("C:/Users/Jaehong/Downloads/mnist_test.txt", true);
        System.out.println(new Date());

        LayerConf conv1 = new LayerConf(LayerType.CONVOLUTION)
        .set("num_output", 20)
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
        .set("num_output", 50)
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
                .setInputDim(new int[]{1, 1, 28, 28})
                .setOutputDim(new int[]{10})
                .addLayer(conv1)
                .addLayer(pool1)
                .addLayer(conv2)
                .addLayer(pool2)
                .addLayer(full1)
                .addLayer(full2)
                .build();

        NeuralNetRunner driver = new NeuralNetRunner(net).setIterations(numIteration)
                                                         .setMiniBatchSize(minibatch);

        System.out.println("Start Learning...");
        Date startTime = new Date();
        driver.train(training_data);
        Date endTime = new Date();

        System.out.println(String.format("Accuracy: %f %%", driver.printAccuracy(test_data)));

        long time = endTime.getTime() - startTime.getTime();
        System.out.println(String.format("Training time: %f secs", (double) time / 1000));
    }


}
