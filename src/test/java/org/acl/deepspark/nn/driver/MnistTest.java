package org.acl.deepspark.nn.driver;

import org.acl.deepspark.data.Sample;
import org.acl.deepspark.nn.conf.LayerConf;
import org.acl.deepspark.nn.conf.NeuralNetConf;
import org.acl.deepspark.nn.functions.ActivatorType;
import org.acl.deepspark.nn.layers.LayerType;
import org.acl.deepspark.utils.CIFARLoader;
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
        LayerConf layer1 = new LayerConf(LayerType.CONVOLUTION);
        layer1.set("numFilters", 20);
        layer1.set("filterRow", 5);
        layer1.set("filterCol", 5);
        layer1.set("stride", 1);
        layer1.set("zeroPad", 0);
        layer1.set("activator", ActivatorType.RECTIFIED_LINEAR);

        LayerConf layer2 = new LayerConf(LayerType.POOLING);
        layer2.set("poolRow", 2);
        layer2.set("poolCol", 2);
        layer2.set("stride", 2);
        layer2.set("activator", ActivatorType.NONE);

        LayerConf layer3 = new LayerConf(LayerType.CONVOLUTION);
        layer3.set("numFilters", 50);
        layer3.set("filterRow", 5);
        layer3.set("filterCol", 5);
        layer3.set("stride", 1);
        layer3.set("zeroPad", 0);
        layer3.set("activator", ActivatorType.RECTIFIED_LINEAR);

        LayerConf layer4 = new LayerConf(LayerType.POOLING);
        layer4.set("poolRow", 2);
        layer4.set("poolCol", 2);
        layer4.set("stride", 2);
        layer4.set("activator", ActivatorType.NONE);
/*
        LayerConf layer5 = new LayerConf(LayerType.CONVOLUTION);
        layer5.set("numFilters", 64);
        layer5.set("filterRow", 5);
        layer5.set("filterCol", 5);
        layer5.set("stride", 1);
        layer5.set("zeroPad", 1);
        layer5.set("activator", ActivatorType.RECTIFIED_LINEAR);

        LayerConf layer6 = new LayerConf(LayerType.POOLING);
        layer6.set("poolRow", 3);
        layer6.set("poolCol", 3);
        layer6.set("stride", 2);
        layer6.set("activator", ActivatorType.NONE);
*/

		LayerConf layer6 = new LayerConf(LayerType.FULLYCONN);
		layer6.set("numNodes", 500);
		layer6.set("activator", ActivatorType.RECTIFIED_LINEAR);

        LayerConf layer7 = new LayerConf(LayerType.FULLYCONN);
        layer7.set("numNodes", 10);
        layer7.set("activator", ActivatorType.SOFTMAX);

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
//                .addLayer(layer5)
                .addLayer(layer6)
                .addLayer(layer7)
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
