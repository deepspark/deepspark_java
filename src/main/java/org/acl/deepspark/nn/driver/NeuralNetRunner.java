package org.acl.deepspark.nn.driver;

import org.acl.deepspark.data.Accumulator;
import org.acl.deepspark.data.Sample;
import org.acl.deepspark.data.Tensor;
import org.acl.deepspark.data.Weight;
import org.acl.deepspark.utils.ArrayUtils;
import org.jblas.util.Random;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Created by Jaehong on 2015-07-16.
 */
public class NeuralNetRunner {
    private NeuralNet net;
    private Accumulator weightAccum;

    private int iteration;
    private int batchSize;

    public NeuralNetRunner(NeuralNet net) {
        this.net = net;
        this.weightAccum = new Accumulator(net.getNumLayers());
    }

    public NeuralNetRunner setIterations(int iteration) {
        this.iteration = iteration;
        return this;
    }

    public NeuralNetRunner setMiniBatchSize(int batchSize) {
        this.batchSize = batchSize;
        return this;
    }

    public void train(Sample[] data) throws Exception {
        int dataSize = data.length;
        for (int i = 0 ; i < iteration; i++) {
        	System.out.println(String.format("%d(th) iteration...", i + 1));

            for (int j = 0; j < batchSize; j++) {
                weightAccum.accumulate(net.train(data[Random.nextInt(dataSize)]));
            }
            net.updateWeight(weightAccum.getAverage());
            weightAccum.clear();
        }
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
            if (ArrayUtils.argmax(sample.label) == ArrayUtils.argmax(output))
                count++;
        }
        return (double) count / data.length * 100;
    }

//    public double printAvgCost(Sample[] data) {
//        INDArray ret
//        INDArray ret;
//        for (Sample sample : data) {
//            net.predict(sample).subi(sample.label)
//        }
//    }
}
