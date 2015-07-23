package org.acl.deepspark.driver;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Date;
import java.util.List;

import org.acl.deepspark.data.Accumulator;
import org.acl.deepspark.data.Sample;
import org.acl.deepspark.nn.conf.spark.DistNeuralNetConfiguration;
import org.acl.deepspark.nn.layers.BaseLayer;
import org.acl.deepspark.nn.layers.ConvolutionLayer;
import org.acl.deepspark.nn.layers.PoolingLayer;
import org.acl.deepspark.utils.MnistLoader;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.storage.StorageLevel;
import org.jblas.DoubleMatrix;


public class DistNeuralNetConfigurationTest implements Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 8811812248690041287L;
	
	public static final int nTest = 10000;
	public static final int minibatch = 100;
	public static final double momentum = 0.5;
	public static final int epoch = 3;
	
	public static void main(String[] args) {
		SparkConf conf = new SparkConf().setAppName("DeepSpark CNN Test Driver").set("spark.serializer", "org.apache.spark.serializer.KryoSerializer");
		Class[] reg_classes = {BaseLayer.class, ConvolutionLayer.class, PoolingLayer.class,
				FullyConnLayer.class, Accumulator.class, Sample.class};
		conf.registerKryoClasses(reg_classes);
		JavaSparkContext sc = new JavaSparkContext(conf);
		
		System.out.println("Data Loading...");
				
		Sample[] train_samples = MnistLoader.loadFromHDFS("mnist_data/train_data.txt",true);
		Sample[] test_samples = MnistLoader.loadFromHDFS("mnist_data/test_data.txt",true);
		
		Collections.shuffle(Arrays.asList(train_samples));
		
		System.out.println(String.format("%d samples loaded...", train_samples.length));
		
		// configure network
		List<BaseLayer> layerList = new ArrayList<BaseLayer>();
		layerList.add(new ConvolutionLayer(9, 9, 20,momentum, 1e-5)); // conv with 20 filters (9x9)
		layerList.add(new PoolingLayer(2)); // max pool
		layerList.add(new FullyConnLayer(10,momentum, 1e-5)); // output
		
		DistNeuralNetConfiguration net = new DistNeuralNetConfiguration();
		
		int[] dim = new int[3];
		dim[0] = train_samples[0].data[0].getRows();
		dim[1] = train_samples[0].data[0].getColumns();
		dim[2] = train_samples[0].data.length;
		net.prepareForTraining(layerList, dim);
		
		// prepare RDD
		
		int numMinibatch = (int) Math.ceil((double) train_samples.length / minibatch);
		double[] batchWeight = new double[numMinibatch];
		for(int i = 0; i < numMinibatch; i++)
			batchWeight[i] = 1.0 / numMinibatch;		
		JavaRDD<Sample>[] rddMinibatch = sc.parallelize(Arrays.asList(train_samples)).persist(StorageLevel.MEMORY_ONLY_SER()).randomSplit(batchWeight);
		
		System.out.println("Start Learning...");
		Date startTime = new Date();
		for(int i = 0 ; i < epoch ; i++) {
			System.out.println(String.format("%d epoch...", i+1));
			for(int j = 0; j < rddMinibatch.length; j++) {
				System.out.println(String.format("%d - epoch, %d minibatch",i+1, j + 1));
				net.training(rddMinibatch[j],minibatch, sc);
			}
		}
		sc.close();
		
		Date endTime = new Date();
		long time = endTime.getTime() - startTime.getTime();
		
		System.out.println(String.format("Training time: %f secs", (double) time / 1000));
			
		DoubleMatrix[] matrix;
		
		System.out.println(String.format("Testing... with %d samples...", nTest));
		int count = 0;
		for(int j = 0 ; j < nTest; j++) {
			matrix = test_samples[j].data;
			if(test_samples[j].label.argmax() == net.getOutput(matrix)[0].argmax())
				count++;
		}
		System.out.println(String.format("Accuracy: %f %%", (double) count / nTest * 100));
	} 
}
