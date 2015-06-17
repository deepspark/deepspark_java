package org.acl.deepspark.nn.conf;

import java.util.Arrays;
import java.util.Collections;
import java.util.Date;

import org.acl.deepspark.data.Sample;
import org.acl.deepspark.nn.layers.FullyConnLayer;
import org.acl.deepspark.nn.layers.cnn.ConvolutionLayer;
import org.acl.deepspark.nn.layers.cnn.PoolingLayer;
import org.acl.deepspark.utils.MnistLoader;
import org.jblas.DoubleMatrix;


public class NeuralNetConfigurationTest {
	
	public static final int nTest = 10000;
	public static final int minibatch = 1;
	
	public static void main(String[] args) {
		System.out.println("Data Loading...");
		
		Sample[] train_data = MnistLoader.loadIntoSamples("C:/Users/Hanjoo Kim/Downloads/mnist_train.txt",true);
		Sample[] test_data = MnistLoader.loadIntoSamples("C:/Users/Hanjoo Kim/Downloads/mnist_test.txt",true);
		
		System.out.println("Shuffling...");
		Collections.shuffle(Arrays.asList(train_data));
		Collections.shuffle(Arrays.asList(test_data));
		
		// configure network
		NeuralNetConfiguration net = new NeuralNetConfiguration(0.1, 1, minibatch,true);
		net.addLayer(new ConvolutionLayer(9, 9, 20,0.5, 1e-5)); // conv with 20 filters (9x9)
		net.addLayer(new PoolingLayer(2)); // max pool
		net.addLayer(new FullyConnLayer(500,0.5, 1e-5)); // hidden
		net.addLayer(new FullyConnLayer(10,0.5, 1e-5)); // output
		
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
