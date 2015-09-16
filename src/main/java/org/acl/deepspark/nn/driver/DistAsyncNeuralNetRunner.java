package org.acl.deepspark.nn.driver;

import org.acl.deepspark.data.Accumulator;
import org.acl.deepspark.data.Sample;
import org.acl.deepspark.data.Tensor;
import org.acl.deepspark.nn.async.ParameterClient;
import org.acl.deepspark.nn.async.ParameterServer;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.VoidFunction;

import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

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
    private int[] port;

    public DistAsyncNeuralNetRunner(NeuralNet net, String host, int[] port) {
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

    public void train(JavaRDD<Sample> data) throws IOException {
        System.out.println("Start async learning...");
        System.out.println(String.format("batchSize: %d", batchSize));
        System.out.println(String.format("iterations: %d", iteration));
        System.out.println(String.format("learningRate: %4f", net.learningRate));
        System.out.println(String.format("momentum: %4f", net.momentum));
        System.out.println(String.format("decayLambda: %4f", net.decayLambda));
        System.out.println(String.format("dropOutRate: %4f", net.dropOutRate));

        int numPartition = (int) data.cache().count() / batchSize;

        ParameterServer server = new ParameterServer(net, batchSize, port);
        server.startServer();

        data.foreachPartition(new VoidFunction<Iterator<Sample>>() {
            private static final long serialVersionUID = -7223288722205378737L;

            @Override
            public void call(Iterator<Sample> samples) throws Exception {
                Accumulator w = new Accumulator(net.getNumLayers());
                List<Sample> sampleList = new ArrayList<Sample>();
                while (samples.hasNext()) {
                    sampleList.add(samples.next());
                }

                for (int i = 0; i < iteration; i++) {
                    System.out.println(String.format("%d th iteration", i));

                    Iterator<Sample> iter = sampleList.iterator();
                    net.setWeights(ParameterClient.getWeights(host, port[1]));
                    while (iter.hasNext()) {
                        w.accumulate(net.train(iter.next()));

                        if (w.getCount() == batchSize) {
                            ParameterClient.sendDelta(host, port[0], w.getAverage());
                            net.setWeights(ParameterClient.getWeights(host, port[1]));
                            w.clear();
                        }
                    }
                    if (w.getCount() != 0) {
                        ParameterClient.sendDelta(host, port[0], w.getAverage());
                        w.clear();
                    }
                }
            }
        });

        server.stopServer();
    }

    public Tensor[] predict(Sample[] data) {
        Tensor[] output = new Tensor[data.length];
        for (int i = 0 ; i < data.length ; i++)
            output[i] = predict(data[i]);
        return output;
    }

    public Tensor predict(Sample data) {
        return net.predict(data);
    }

    public double printAccuracy(Sample[] data) {
        int count = 0;
        for (Sample sample : data) {
            Tensor output = net.predict(sample);
            if (sample.label.slice(0,0).argmax() == output.slice(0,0).argmax())
                count++;
        }
        return (double) count / data.length * 100;
    }

}
