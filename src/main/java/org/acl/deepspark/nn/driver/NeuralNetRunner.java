package org.acl.deepspark.nn.driver;

import org.acl.deepspark.data.Accumulator;
import org.acl.deepspark.data.Sample;
import org.jblas.util.Random;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Created by Jaehong on 2015-07-16.
 */
public class NeuralNetRunner {
    private NeuralNet net;
    private INDArray[] deltaWeights;
    private Accumulator weightAccum;

    private int iteration;
    private int batchSize;
    private double learningRate;
    private double decayLambda;
    private double momentum;

    public NeuralNetRunner(NeuralNet net) {
        this.net = net;
        this.weightAccum = new Accumulator(net.getNumLayers());
        /* default configuration */

        this.iteration = 10000;
        this.batchSize = 1;
        this.learningRate = 0.1;
    }

    public NeuralNetRunner setIterations(int iteration) {
        this.iteration = iteration;
        return this;
    }

    public NeuralNetRunner setMiniBatchSize(int batchSize) {
        this.batchSize = batchSize;
        return this;
    }

    public NeuralNetRunner setLearningRate(double learningRate) {
        this.learningRate = learningRate;
        return this;
    }

    public NeuralNetRunner setDecayLambda(double decayLambda) {
        this.de
    }

    learningRate = conf.getParams().get("learningRate");
    decayLambda = conf.getParams().get("decayLambda");
    momentum = conf.getParams().get("momentum");
    dropOutRate = conf.getParams().get("dropOutRate");

    public void train(Sample[] data) throws Exception {
        int dataSize = data.length;
        int index;
        for (int i = 0 ; i < iteration; i++) {
            for (int j = 0; j < batchSize; j++) {
                index = Random.nextInt(dataSize);
                weightAccum.accumulate(net.train(data[index]));
            }
            net.updateWeight( weightAccum.getAverage());
        }
    }

    public INDArray[] predict(Sample[] data) {
        if (data != null) {
            INDArray[] output = new INDArray[data.length];
            for (int i = 0 ; i < data.length ; i++)
                output[i] = predict(data[i]);
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
