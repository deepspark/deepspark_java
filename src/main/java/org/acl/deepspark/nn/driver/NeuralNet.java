package org.acl.deepspark.nn.driver;

import org.acl.deepspark.data.Accumulator;
import org.acl.deepspark.data.Sample;
import org.acl.deepspark.data.Weight;
import org.acl.deepspark.nn.conf.LayerConf;
import org.acl.deepspark.nn.conf.NeuralNetConf;
import org.acl.deepspark.nn.functions.ActivatorType;
import org.acl.deepspark.nn.layers.ConvolutionLayer;
import org.acl.deepspark.nn.layers.FullyConnectedLayer;
import org.acl.deepspark.nn.layers.Layer;
import org.acl.deepspark.nn.layers.PoolingLayer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;

/**
 * Created by Jaehong on 2015-07-16.
 */
public class NeuralNet {
    private Layer[]     layers;
    private Weight[]    weights;
    private Weight[]    weightUpdates;

    double learningRate;
    double decayLambda;
    double momentum;
    double dropOutRate;

    public NeuralNet(final NeuralNetConf conf) {
        learningRate = conf.getLearningRate();
        decayLambda = conf.getDecayLambda();
        momentum = conf.getMomentum();
        dropOutRate = conf.getDropOutRate();
        initNetwork(conf);
    }

    public void initNetwork(final NeuralNetConf conf) {
        int size = conf.getLayerList().size();
        layers = new Layer[size];
        weights = new Weight[size];
        weightUpdates = new Weight[size];
        buildNetwork(conf.getLayerList(), conf.getDimIn());
    }

    private void buildNetwork(ArrayList<LayerConf> arr, int[] dimIn) {
        for (int i = 0 ; i< arr.size(); i++) {
            LayerConf layerConf = arr.get(i);
            ActivatorType activator = layerConf.getActivator();
            Layer layer = null;

            switch (layerConf.getType()) {
                case CONVOLUTION:
                    layer = new ConvolutionLayer(activator);
                    break;
                case POOLING:
                    layer = new PoolingLayer(activator);
                    break;
                case FULLYCONN:
                    layer = new FullyConnectedLayer(activator);
                    break;
            }
            if (layer != null) {
                layers[i] = layer;

                weights[i] = layers[i].createWeight(layerConf, dimIn);
                weightUpdates[i] = new Weight(weights[i].getShape(), dimIn);
            }
        }
    }

    public void setWeights(Weight[] weights) {
        this.weights = weights;
    }

    public Weight[] getWeights() {
        return weights;
    }

    public int getNumLayers() {
        return layers.length;
    }

    public INDArray feedForward(Sample in) {
        INDArray activationOut = in.getFeature();
        for (int i = 0; i < layers.length; i++)
            activationOut = layers[i].generateOutput(weights[i], activationOut);
        return activationOut;
    }

    public Weight[] train(Sample in) {
        Weight[] gradient = new Weight[layers.length];
        INDArray[] preActivation = new INDArray[layers.length];
        INDArray[] postActivation = new INDArray[layers.length + 1];
        postActivation[0] = in.data;

        for (int i = 0; i < layers.length; i++) {
            preActivation[i] = layers[i].generateOutput(weights[i], postActivation[i]);
            postActivation[i+1] = layers[i].activate(preActivation[i]);
        }

        INDArray error = postActivation[layers.length].sub(in.label);
        for (int i = layers.length-1; i >= 0; i--) {
            gradient[i] = layers[i].gradient(postActivation[i], error);
            if (i > 0)
                error = layers[i].deriveDelta(weights[i], preActivation[i-1], error);
        }
        return gradient;
    }

    public void updateWeight(Weight[] deltaWeight) throws Exception {
        if (weights.length != deltaWeight.length)
            throw new Exception("weight dimension mismatch");
        for (int i = 0 ; i < weights.length; i++) {
            weightUpdates[i].muli(momentum)
                            .subi(weights[i].mul(learningRate*decayLambda))
                            .subi(deltaWeight[i].mul(learningRate));
            weights[i].addi(weightUpdates[i]);
        }
    }
}
