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

import java.util.Iterator;

/**
 * Created by Jaehong on 2015-07-31.
 */
public class DistNeuralNetRunner {

    private JavaSparkContext sc;
    private NeuralNet net;


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
/*
        for (int i = 0 ; i < iteration; i++) {
            System.out.println(String.format("%d(th) iteration...", i + 1));

            for (int j = 0; j < batchSize; j++) {
                weightAccum.accumulate(net.train(data[Random.nextInt(dataSize)]));
            }
            net.updateWeight(weightAccum.getAverage());
            weightAccum.clear();
        }
*/
        System.out.println("Start learning...");
        System.out.println(String.format("Partitioning into %d pieces", (int) data.count() / batchSize));

        int numPartition = (int) data.count() / batchSize;
        double[] weights = new double[numPartition];
        for (int i = 0 ; i < numPartition; i++) {
            weights[i] = 1.0;
        }

        JavaRDD<Sample>[] partition = data.randomSplit(weights);
        for (JavaRDD<Sample> rdd : partition) {
            rdd.cache();
        }

        final Accumulator<Weight[]> weightAccumulator = sc.accumulator(net.getWeights(), new DistAccumulator());
        Broadcast<NeuralNet> broadcastNet = sc.broadcast(net);




        // getting output;
        JavaRDD<DoubleMatrix> delta = data.map(new Function<Sample, DoubleMatrix>() {
            @Override
            public DoubleMatrix call(Sample v1) throws Exception {
                BaseLayer[] layerList = layers.value();
                DoubleMatrix[] output = v1.data;
                for(int l = 0; l < layerList.length; l++) {
                    BaseLayer a = layerList[l];
                    a.setInput(output);
                    output = a.getOutput();
                }
                DoubleMatrix error = output[0].sub(v1.label);
                System.out.println(error.sum() / error.length);

                return error;
            }
        });

        //backpropagation
        JavaRDD<Accumulator> dWeight = delta.map(new Function<DoubleMatrix, Accumulator>() {
            @Override
            public Accumulator call(DoubleMatrix arg0) throws Exception {
                BaseLayer[] layerList = layers.value();
                DoubleMatrix[] error = new DoubleMatrix[1];
                error[0] = arg0;

                Accumulator deltas = new Accumulator(layerList.length);

                // Back-propagation
                for(int l = layerList.length - 1; l >=0 ; l--) {
                    BaseLayer a =layerList[l];
                    a.setDelta(error);

                    deltas.gradWList[l] = a.deriveGradientW();

                    if(a.getDelta() != null) {
                        double[] b = new double[a.getDelta().length];
                        for(int i = 0; i < a.getDelta().length; i++)
                            b[i] = a.getDelta()[i].sum();
                        deltas.gradBList[l] = b;
                    }

                    error = a.deriveDelta();
                }

                return deltas;
            }
        });

        Accumulator gradient = dWeight.fold(getEmptyDeltaWeight(), new Function2<Accumulator, Accumulator, Accumulator>() {
            @Override
            public Accumulator call(Accumulator v1, Accumulator v2) throws Exception {
                for(int i1 = 0; i1 < v1.gradWList.length; i1++) {
                    if(v1.gradWList[i1] != null && v2.gradWList[i1] != null) {
                        for(int j1 = 0; j1 < v1.gradWList[i1].length; j1++) {
                            for(int k = 0; k < v1.gradWList[i1][j1].length;k++) {
                                v1.gradWList[i1][j1][k].addi(v2.gradWList[i1][j1][k]);
                            }
                        }

                        for(int j1 = 0; j1 < v1.gradBList[i1].length; j1++) {
                            v1.gradBList[i1][j1] += v2.gradBList[i1][j1];
                        }
                    }
                }

                return v1;
            }
        });

        //update
        update(gradient, minibatchSize);
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
