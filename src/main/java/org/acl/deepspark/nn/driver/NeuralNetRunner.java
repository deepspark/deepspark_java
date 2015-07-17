package org.acl.deepspark.nn.driver;

import org.acl.deepspark.data.Sample;

/**
 * Created by Jaehong on 2015-07-16.
 */
public class NeuralNetRunner {
    private NeuralNet net;
    private int iteration;
    private int batchSize;

    public NeuralNetRunner(NeuralNet net) {
        this.net = net;
    }

    public NeuralNetRunner setIterations(int iteration) {
        this.iteration = iteration;
        return this;
    }

    public NeuralNetRunner setMiniBatchSize(int batchSize) {
        this.batchSize = batchSize;
        return this;
    }

    public void train(Sample[] data) {
        for (int i = 0 ; i < iteration) {

        }
    }

    public void predict(Sample[] data) {

    }
}
