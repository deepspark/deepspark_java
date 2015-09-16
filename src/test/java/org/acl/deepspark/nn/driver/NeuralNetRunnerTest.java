package org.acl.deepspark.nn.driver;

import org.acl.deepspark.data.Sample;
import org.acl.deepspark.nn.conf.LayerConf;
import org.acl.deepspark.nn.conf.NeuralNetConf;
import org.acl.deepspark.nn.functions.ActivatorType;
import org.acl.deepspark.nn.layers.LayerType;
import org.acl.deepspark.utils.CIFARLoader;
import org.acl.deepspark.utils.MnistLoader;

import java.util.Date;


public class NeuralNetRunnerTest {

	public static final int minibatch = 100;
	public static final int numIteration = 600;

	public static final double learningRate = 0.1;
	public static final double decayLambda = 0.0005;
	public static final double momentum = 0.5;
	public static final double dropOut = 0.0;
	
	public static void main(String[] args) throws Exception {

		Sample[] training_data = CIFARLoader.loadIntoSamples("C:/Users/Jaehong/Downloads/train_batch.bin", true);
		Sample[] test_data = CIFARLoader.loadIntoSamples("C:/Users/Jaehong/Downloads/test_batch.bin", true);
		System.out.println(new Date());
		LayerConf layer1 = new LayerConf(LayerType.CONVOLUTION);
		layer1.set("numFilters", 64);
		layer1.set("filterRow", 5);
		layer1.set("filterCol", 5);
		layer1.set("activator", ActivatorType.RECTIFIED_LINEAR);

		LayerConf layer2 = new LayerConf(LayerType.POOLING);
		layer2.set("poolRow", 2);
		layer2.set("poolCol", 2);
		layer2.set("activator", ActivatorType.NONE);

		LayerConf layer3 = new LayerConf(LayerType.CONVOLUTION);
		layer3.set("numFilters", 64);
		layer3.set("filterRow", 5);
		layer3.set("filterCol", 5);
		layer3.set("activator", ActivatorType.RECTIFIED_LINEAR);

		LayerConf layer4 = new LayerConf(LayerType.POOLING);
		layer4.set("poolRow", 2);
		layer4.set("poolCol", 2);
		layer4.set("activator", ActivatorType.NONE);

		LayerConf layer5 = new LayerConf(LayerType.CONVOLUTION);
		layer5.set("numFilters", 64);
		layer5.set("filterRow", 5);
		layer5.set("filterCol", 5);
		layer5.set("activator", ActivatorType.RECTIFIED_LINEAR);
/*
		LayerConf layer6 = new LayerConf(LayerType.POOLING);
		layer6.set("poolRow", 2);
		layer6.set("poolCol", 2);
		layer6.set("activator", ActivatorType.NONE);
*/
		LayerConf layer6 = new LayerConf(LayerType.FULLYCONN);
		layer6.set("numNodes", 200);
		layer6.set("activator", ActivatorType.RECTIFIED_LINEAR);

		LayerConf layer7 = new LayerConf(LayerType.FULLYCONN);
		layer7.set("numNodes", 10);
		layer7.set("activator", ActivatorType.SOFTMAX);

		NeuralNet net = new NeuralNetConf()
							.setLearningRate(learningRate)
							.setDecayLambda(decayLambda)
							.setMomentum(momentum)
							.setDropOutRate(dropOut)
							.setInputDim(new int[]{3, 32, 32})
							.setOutputDim(new int[]{10})
							.addLayer(layer1)
							.addLayer(layer2)
							.addLayer(layer3)
							.addLayer(layer4)
							.addLayer(layer5)
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
