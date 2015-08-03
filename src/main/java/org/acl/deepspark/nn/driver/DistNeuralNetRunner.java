package org.acl.deepspark.nn.driver;

import org.acl.deepspark.data.Sample;
import org.acl.deepspark.data.Weight;
import org.acl.deepspark.data.DistAccumulator;
import org.acl.deepspark.utils.ArrayUtils;
import org.apache.spark.Accumulator;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.broadcast.Broadcast;
import org.jblas.util.Random;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;

/**
 * Created by Jaehong on 2015-07-31.
 */
public class DistNeuralNetRunner implements Serializable {

    private JavaSparkContext sc;
    private NeuralNet net;
    private Weight[] weight;

    private int iteration;
    private int batchSize;

    public DistNeuralNetRunner(JavaSparkContext sc, NeuralNet net) {

        this.sc = sc;
        this.net = net;

        /** default configuration **/
        iteration = 0;
        batchSize = 1;
    }

    public DistNeuralNetRunner setIterations(int iteration) {
        this.iteration = iteration;
        return this;
    }

    public DistNeuralNetRunner setMiniBatchSize(int batchSize) {
        this.batchSize = batchSize;
        return this;
    }

    public void train(JavaRDD<Sample> data) throws Exception {

        System.out.println("Start learning...");
        System.out.println(String.format("Partitioning into %d pieces", (int) data.count() / batchSize));

        int numPartition = (int) data.count() / batchSize;
        double[] weights = new double[numPartition];
        for (int i = 0 ; i < numPartition; i++) {
            weights[i] = 1.0;
        }

        JavaRDD<Sample>[] partition = data.cache().randomSplit(weights);

        Weight[] init = new Weight[net.getWeights().length];
        for (int i = 0 ; i < net.getWeights().length; i++) {
            init[i] = new Weight(net.getWeights()[i].getWeightShape(), net.getWeights()[i].getBiasShape());
        }

        final Accumulator<Weight[]> deltaAccum = sc.accumulator(init, new DistAccumulator());

        for (int i = 0 ; i < iteration; i++) {
            final Broadcast<Weight[]> broadcast = sc.broadcast(net.getWeights());
            JavaRDD<Sample> rdd = partition[Random.nextInt(numPartition)]; // Test needed
            /** TODO: check whether net is updated for each iterations */

            rdd.foreach(new VoidFunction<Sample>() {
                @Override
                public void call(Sample sample) throws Exception {
                    net.setWeights(broadcast.getValue());
                    deltaAccum.add(net.train(sample));
                }
            });

            Weight[] delta = deltaAccum.value();
            for (int j = 0; j < delta.length; j++) {
                delta[i].divi(batchSize);
            }
            net.updateWeight(delta);
            deltaAccum.zero();
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
