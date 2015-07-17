package org.acl.deepspark.nn.driver;

import org.acl.deepspark.data.Sample;
import org.jblas.util.Random;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Created by Jaehong on 2015-07-16.
 */
public class NeuralNetRunner {
    private NeuralNet net;
    private int iteration;
    private int batchSize;

    public NeuralNetRunner(NeuralNet net) {
        this.net = net;

        /* default configuration */
        this.iteration = 10000;
        this.batchSize = 1;
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
        int dataSize = data.length;
        INDArray error;
        for (int i = 0 ; i < iteration; i++) {
            for (int j = 0; j < batchSize; j++) {
                int index = Random.nextInt(dataSize);
                error = data[index].label.sub(net.feedForward(data[index]));
                net.backPropagate(error);
            }
            net.updateWeight();
        }
    }

    public INDArray[] predict(Sample[] data) {
        if (data != null) {
            INDArray[] output = new INDArray[data.length];
            for (int i = 0 ; i < data.length ; i++) {
                output[i] = predict(data[i]);
            }
            return output;
        }
        return null;
    }

    public INDArray predict(Sample data) {
        if (data != null)
            return net.feedForward(data);
        return null;
    }
}
