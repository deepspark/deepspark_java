package org.acl.deepspark.nn.conf;

import org.acl.deepspark.nn.layers.FullyConnLayer;
import org.acl.deepspark.nn.layers.cnn.ConvolutionLayer;
import org.acl.deepspark.nn.layers.cnn.PoolingLayer;
import org.acl.deepspark.utils.MnistLoader;
import org.jblas.DoubleMatrix;
import org.jblas.util.Random;


public class NeuralNetConfigurationTest {
	
	public static void main(String[] args) {
		DoubleMatrix[] data = MnistLoader.loadData("C:\\Users\\Jaehong\\Downloads\\mnist\\mnist_train.txt");
		DoubleMatrix[] label = MnistLoader.loadLabel("C:\\Users\\Jaehong\\Downloads\\mnist\\mnist_train.txt");
		
		NeuralNetConfiguration net = new NeuralNetConfiguration(120000);
		net.addLayer(new ConvolutionLayer(3, 3, 2));
		net.addLayer(new PoolingLayer(2));
		net.addLayer(new FullyConnLayer(10));
		
		net.training(data, label);
		
		DoubleMatrix[] matrix = new DoubleMatrix[1];
		
		for(int i = 0 ; i < 20; i++) {
			System.out.println(String.valueOf(i) + "th data");
			matrix[0] = data[i];
			System.out.println(label[i]);
			System.out.println(net.getOutput(matrix)[0]);
		}
				
	} 
}
