package org.acl.deepspark.nn.conf;

import java.util.Date;

import org.acl.deepspark.nn.layers.FullyConnLayer;
import org.acl.deepspark.nn.layers.cnn.ConvolutionLayer;
import org.acl.deepspark.nn.layers.cnn.PoolingLayer;
import org.acl.deepspark.utils.MnistLoader;
import org.jblas.DoubleMatrix;


public class NeuralNetConfigurationTest {
	
	public static void main(String[] args) {
		DoubleMatrix[] train_data = MnistLoader.loadData("/Users/kprotoss/Downloads/mnist_train.txt");
		DoubleMatrix[] train_label = MnistLoader.loadLabel("/Users/kprotoss/Downloads/mnist_train.txt");
		
		DoubleMatrix[] test_data = MnistLoader.loadData("/Users/kprotoss/Downloads/mnist_test.txt");
		DoubleMatrix[] test_label = MnistLoader.loadLabel("/Users/kprotoss/Downloads/mnist_test.txt");
		
		
		NeuralNetConfiguration net = new NeuralNetConfiguration(0.1, 3, 100);
		net.addLayer(new ConvolutionLayer(5, 5, 10));
		net.addLayer(new PoolingLayer(2));
		net.addLayer(new FullyConnLayer(100));
		net.addLayer(new FullyConnLayer(10));

		
		Date startTime = new Date();
		net.training(train_data, train_label);
		DoubleMatrix[] matrix = new DoubleMatrix[1];
		
		int count = 0;
		int ITERATION = 10000;
		for(int j = 0 ; j < ITERATION; j++) {
			matrix[0] = test_data[j];
			if(test_label[j].argmax() == net.getOutput(matrix)[0].argmax())
				count++;
		}
		
		System.out.println(String.format("Accuracy: %f %%", (double) count / ITERATION * 100));
		Date endTime = new Date();
		
		long time = endTime.getTime() - startTime.getTime();
		
		
		
		/*
		 * DoubleMatrix[] matrix = new DoubleMatrix[1];
		int count = 0;
		int ITERATION = 10000;
		for(int i = 0 ; i < ITERATION; i++) {
			//System.out.println(String.valueOf(i) + "th data");
			matrix[0] = test_data[i];
			if(test_label[i].argmax() == net.getOutput(matrix)[0].argmax())
				count++;
			
			
			System.out.println(test_label[i]);
			System.out.println(net.getOutput(matrix)[0]);
			
		}*/
		System.out.println(String.format("Training time: %f secs", (double) time / 1000));
	} 
}
