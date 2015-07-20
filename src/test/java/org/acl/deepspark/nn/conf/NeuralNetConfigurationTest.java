package org.acl.deepspark.nn.conf;

import java.util.Arrays;
import java.util.Collections;
import java.util.Date;

import org.acl.deepspark.data.Sample;
import org.acl.deepspark.nn.driver.NeuralNet;
import org.acl.deepspark.nn.driver.NeuralNetRunner;
import org.acl.deepspark.nn.functions.Activator;
import org.acl.deepspark.nn.functions.ActivatorType;
import org.acl.deepspark.nn.layers.FullyConnLayer;
import org.acl.deepspark.nn.layers.LayerType;
import org.acl.deepspark.nn.layers.cnn.ConvolutionLayer;
import org.acl.deepspark.nn.layers.cnn.PoolingLayer;
import org.acl.deepspark.utils.MnistLoader;


public class NeuralNetConfigurationTest {
	
	public static final int nTest = 10000;
	public static final int minibatch = 100;
	public static final double momentum = 0.5;
	
	public static void main(String[] args) {
		System.out.println("Data Loading...");
		
		Sample[] train_data = MnistLoader.loadIntoSamples("C:/Users/Hanjoo Kim/Downloads/mnist_train.txt",true);
		Sample[] test_data = MnistLoader.loadIntoSamples("C:/Users/Hanjoo Kim/Downloads/mnist_test.txt",true);
		
		System.out.println("Shuffling...");
		Collections.shuffle(Arrays.asList(train_data));
		Collections.shuffle(Arrays.asList(test_data));


		LayerConf layer1 = new LayerConf(LayerType.CONVOLUTION);
		layer1.setFilterSize(new int[]{3, 3});
		layer1.setNumFilters(10);
		layer1.setActivator(Activator.SIGMOID);

		LayerConf layer2 = new LayerConf(LayerType.POOLING);
		layer2.setPoolingSize(2);
		layer2.setActivator(Activator.SIGMOID);

		LayerConf layer3 = new LayerConf(LayerType.FULLYCONN);
		layer3.setOutputUnit(120);
		layer2.setActivator(Activator.SIGMOID);

		LayerConf layer4 = new LayerConf(LayerConf.FULLYCONN);
		layer3.setOutputUnit(10);

		NeuralNet net = new NeuralNetConf().setLearningRate(0.1)
											.setDecayLambda(0.0001)
											.setMomentum(0.9)
											.setDropOutRate(0.0)
											.setInputDim(new int[]{28, 28, 1})
											.setOutputDim(new int[]{10})
											.addLayer(layer1)
											.addLayer(layer2)
											.addLayer(layer3)
											.addLayer(layer4)
											.build();

		NeuralNetRunner driver = new NeuralNetRunner(net).setIterations(10000)
														 .setMiniBatchSize(10)
														 .;
		driver.train(train_data);
		driver.predict(test_data);

		// configure network
		NeuralNetConf net = new NeuralNetConf(0.1, 1, minibatch,true);
		net.addLayer(new ConvolutionLayer(9, 9, 20,momentum, 1e-5)); // conv with 20 filters (9x9)
		net.addLayer(new PoolingLayer(2)); // max pool
		net.addLayer(new FullyConnLayer(10,momentum, 1e-5)); // output
		
		int[] dim = new int[3];
		dim[0] = train_data[0].data[0].getRows();
		dim[1] = train_data[0].data[0].getColumns();
		dim[2] = train_data[0].data.length;
		
		net.prepareForTraining(dim);
		System.out.println("Start Learning...");
		Date startTime = new Date();
		net.training(train_data);
		
		System.out.println(String.format("Testing... with %d samples...", nTest));
		int count = 0;
		for(int j = 0 ; j < nTest; j++) {
			if(test_data[j].label.argmax() == net.getOutput(test_data[j].data)[0].argmax())
				count++;
		}
		
		System.out.println(String.format("Accuracy: %f %%", (double) count / nTest * 100));
		Date endTime = new Date();
		
		long time = endTime.getTime() - startTime.getTime();
		
		System.out.println(String.format("Training time: %f secs", (double) time / 1000));
	} 
}
