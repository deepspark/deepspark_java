package org.acl.deepspark.nn.driver;

import java.io.Serializable;
import java.util.Iterator;

import org.acl.deepspark.data.Accumulator;
import org.acl.deepspark.data.Sample;
import org.acl.deepspark.data.Weight;
import org.acl.deepspark.nn.async.ParameterClient;
import org.acl.deepspark.utils.ArrayUtils;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.VoidFunction;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Created by Jaehong on 2015-07-31.
 */
public class DistAsyncNeuralNetRunner implements Serializable {

    /**
	 * 
	 */
	private static final long serialVersionUID = -5070368428661536358L;

	private NeuralNet net;

    private int iteration;
    private int batchSize;
    private String host;
    private int port;

    public DistAsyncNeuralNetRunner(NeuralNet net, String host, int port) {
        this.net = net;
        this.host = host;
        this.port = port;
    }

    public DistAsyncNeuralNetRunner setIterations(int iteration) {
        this.iteration = iteration;
        return this;
    }

    public DistAsyncNeuralNetRunner setMiniBatchSize(int batchSize) {
        this.batchSize = batchSize;
        return this;
    }

    public void train(JavaRDD<Sample> data) {
        System.out.println("Start async learning...");
        System.out.println(String.format("batchSize: %d", batchSize));
        System.out.println(String.format("iterations: %d", iteration));
        System.out.println(String.format("learningRate: %4f", net.learningRate));
        System.out.println(String.format("momentum: %4f", net.momentum));
        System.out.println(String.format("decayLambda: %4f", net.decayLambda));
        System.out.println(String.format("dropOutRate: %4f", net.dropOutRate));
        
        Weight[] init = new Weight[net.getWeights().length];
        for (int i = 0 ; i < net.getWeights().length; i++) {
            if (net.getWeights()[i] != null)
                init[i] = new Weight(net.getWeights()[i].getWeightShape(), net.getWeights()[i].getBiasShape());
        }
        
        data.cache();
        
        for (int i = 0 ; i < iteration; i++) {
            data.foreachPartition(new VoidFunction<Iterator<Sample>>() {
                /**
				 * 
				 */
				private static final long serialVersionUID = -7223288722205378737L;

				@Override
                public void call(Iterator<Sample> samples) throws Exception {
					//initialize
					Accumulator w = new Accumulator(net.getNumLayers());
					int count = 0;
					
					while(samples.hasNext()) {
						if(count == 0) {
							w.clear();
						}
						
						Sample sample = samples.next();
						w.accumulate(net.train(sample));
	                    
	                    count++;
	                    if(count == batchSize) {
	                    	ParameterClient.sendDelta(host, port, w.getAverage());
	                    	net.setWeights(ParameterClient.getWeights(host, port));
	                    	
	                    	w.clear();
	                    	count = 0;
	                    }
					}
					
					if(count != 0) {
						ParameterClient.sendDelta(host, port, w.getAverage());
                    	net.setWeights(ParameterClient.getWeights(host, port));
                    	
                    	w.clear();
                    	count = 0;
					}
                }
            });
        }
    }

    public INDArray[] predict(Sample[] data) {
        INDArray[] output = new INDArray[data.length];
        for (int i = 0 ; i < data.length ; i++)
            output[i] = predict(data[i]);
        return output;
    }

    public INDArray predict(Sample data) {
        return net.predict(data);
    }

    public double printAccuracy(Sample[] data) {
        int count = 0;
        for (Sample sample : data) {
            INDArray output = net.predict(sample);
            if (ArrayUtils.argmax(sample.label) == ArrayUtils.argmax(output))
                count++;
        }
        return (double) count / data.length * 100;
    }

}
