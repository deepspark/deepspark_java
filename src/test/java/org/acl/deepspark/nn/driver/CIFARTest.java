package org.acl.deepspark.nn.driver;

import org.acl.deepspark.data.Sample;
import org.acl.deepspark.data.WeightType;
import org.acl.deepspark.nn.conf.LayerConf;
import org.acl.deepspark.nn.conf.NeuralNetConf;
import org.acl.deepspark.nn.functions.ActivatorType;
import org.acl.deepspark.nn.layers.LayerType;
import org.acl.deepspark.utils.CIFARLoader;

import java.util.Date;

/**
 * Created by Jaehong on 2015-10-01.
 */
public class CIFARTest {
    public static final int minibatch = 100;
    public static final int numIteration = 5000;

    public static final double learningRate = 0.001;
    public static final double decayLambda = 0.004;
    public static final double momentum = 0.9;
    public static final double dropOut = 0.0;

    public static void main(String[] args) throws Exception {

        Sample[] training_data = CIFARLoader.loadIntoSamples("C:/Users/Jaehong/Downloads/train_batch.bin", true);
        Sample[] test_data = CIFARLoader.loadIntoSamples("C:/Users/Jaehong/Downloads/test_batch.bin", true);
        System.out.println(new Date());
        LayerConf conv1 = new LayerConf(LayerType.CONVOLUTION)
        .set("num_output", 64)
        .set("kernel_row", 5)
        .set("kernel_col", 5)
        .set("stride", 1)
        .set("zeroPad", 2)
        .set("weight_type", WeightType.GAUSSIAN)
        .set("weight_value", 0.0001f)
        .set("activator", ActivatorType.RECTIFIED_LINEAR);

        LayerConf pool1 = new LayerConf(LayerType.POOLING)
        .set("kernel_row", 2)
        .set("kernel_col", 2)
        .set("stride", 2)
        .set("activator", ActivatorType.NONE);

        LayerConf conv2 = new LayerConf(LayerType.CONVOLUTION)
        .set("num_output", 64)
        .set("kernel_row", 5)
        .set("kernel_col", 5)
        .set("stride", 1)
        .set("zeroPad", 2)
        .set("weight_type", WeightType.GAUSSIAN)
        .set("weight_value", 0.01f)
        .set("activator", ActivatorType.RECTIFIED_LINEAR);

        LayerConf pool2 = new LayerConf(LayerType.POOLING)
        .set("kernel_row", 2)
        .set("kernel_col", 2)
        .set("stride", 2)
        .set("activator", ActivatorType.NONE);

        LayerConf conv3 = new LayerConf(LayerType.CONVOLUTION)
        .set("num_output", 64)
        .set("kernel_row", 5)
        .set("kernel_col", 5)
        .set("stride", 1)
        .set("zeroPad", 2)
        .set("weight_type", WeightType.GAUSSIAN)
        .set("weight_value", 0.01f)
        .set("activator", ActivatorType.RECTIFIED_LINEAR);

        LayerConf pool3 = new LayerConf(LayerType.POOLING)
        .set("kernel_row", 2)
        .set("kernel_col", 2)
        .set("stride", 2)
        .set("activator", ActivatorType.NONE);

        LayerConf full1 = new LayerConf(LayerType.FULLYCONN)
        .set("num_output", 10)
        .set("weight_type", WeightType.GAUSSIAN)
        .set("weight_value", 0.01f)
        .set("activator", ActivatorType.SOFTMAX);

        NeuralNet net = new NeuralNetConf()
                .setLearningRate(learningRate)
                .setDecayLambda(decayLambda)
                .setMomentum(momentum)
                .setDropOutRate(dropOut)
                .setInputDim(new int[]{1, 3, 32, 32})
                .setOutputDim(new int[]{10})
                .addLayer(conv1)
                .addLayer(pool1)
                .addLayer(conv2)
                .addLayer(pool2)
                .addLayer(conv3)
				.addLayer(pool3)
                .addLayer(full1)
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
