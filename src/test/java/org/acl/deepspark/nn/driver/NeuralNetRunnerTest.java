package org.acl.deepspark.nn.driver;

import org.acl.deepspark.data.Sample;
import org.acl.deepspark.nn.conf.LayerConf;
import org.acl.deepspark.nn.conf.NeuralNetConf;
import org.acl.deepspark.nn.driver.NeuralNet;
import org.acl.deepspark.nn.driver.NeuralNetRunner;
import org.acl.deepspark.nn.functions.ActivatorType;


import org.acl.deepspark.nn.layers.LayerType;
import org.acl.deepspark.utils.MnistLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Date;


public class NeuralNetRunnerTest {

	public static final int minibatch = 1;
	public static final int numIteration = 600;

	public static final double learningRate = 0.1;
	public static final double decayLambda = 0.000;
	public static final double momentum = 0.0;
	
	public static void main(String[] args) {

		System.out.println("Data Loading...");
		Sample[] training_data = MnistLoader.loadIntoSamples("C:/Users/Jaehong/Downloads/mnist_train.txt", true);
		Sample[] test_data = MnistLoader.loadIntoSamples("C:/Users/Jaehong/Downloads/mnist_test.txt", true);;

		LayerConf layer1 = new LayerConf(LayerType.CONVOLUTION);
		layer1.set("numFilters", 2);
		layer1.set("filterRow", 3);
		layer1.set("filterCol", 3);
		layer1.set("activator", ActivatorType.SIGMOID);

		LayerConf layer2 = new LayerConf(LayerType.POOLING);
		layer2.set("poolRow", 2);
		layer2.set("poolCol", 2);
		layer2.set("activator", ActivatorType.SIGMOID);

		LayerConf layer3 = new LayerConf(LayerType.FULLYCONN);
		layer3.set("numNodes", 120);
		layer3.set("activator", ActivatorType.SIGMOID);

		LayerConf layer4 = new LayerConf(LayerType.FULLYCONN);
		layer4.set("numNodes", 10);
		layer4.set("activator", ActivatorType.SIGMOID);

		NeuralNet net = new NeuralNetConf()
							.setLearningRate(learningRate)
							.setDecayLambda(decayLambda)
							.setMomentum(momentum)
							.setDropOutRate(0.0)
							.setInputDim(new int[]{1, 28, 28})
							.setOutputDim(new int[]{10})
							.addLayer(layer1)
							.addLayer(layer2)
							.addLayer(layer3)
							.addLayer(layer4)
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
