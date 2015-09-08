package org.acl.deepspark.nn.driver;

import org.acl.deepspark.data.Sample;
import org.acl.deepspark.data.Tensor;
import org.acl.deepspark.data.Weight;
import org.acl.deepspark.nn.conf.LayerConf;
import org.acl.deepspark.nn.conf.NeuralNetConf;
import org.acl.deepspark.nn.layers.ConvolutionLayer;
import org.acl.deepspark.nn.layers.FullyConnectedLayer;
import org.acl.deepspark.nn.layers.Layer;
import org.acl.deepspark.nn.layers.PoolingLayer;

import java.io.Serializable;
import java.util.ArrayList;

/**
 * Created by Jaehong on 2015-07-16.
 */
public class NeuralNet implements Serializable {
    public double learningRate;
    public double decayLambda;
    public double momentum;
    public double dropOutRate;

    private Layer[]     layers;
    private Weight[]    weights;
    private Weight[]    weightUpdates;

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
            switch (layerConf.getType()) {
                case CONVOLUTION:
                    layers[i] = new ConvolutionLayer(dimIn, layerConf);
                    break;
                case POOLING:
                    layers[i] = new PoolingLayer(dimIn, layerConf);
                    break;
                case FULLYCONN:
                    layers[i] = new FullyConnectedLayer(dimIn, layerConf);
                    break;
            }
            weights[i] = layers[i].createWeight(layerConf, dimIn);
            dimIn = layers[i].calculateOutputDimension();
            if (weights[i] != null)
                weightUpdates[i] = new Weight(weights[i].getWeightShape(), weights[i].getBiasShape());
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

    public Weight[] train(Sample in) throws Exception {
        Weight[] gradient = new Weight[layers.length];
        Tensor[] output = new Tensor[layers.length];
        Tensor[] activated = new Tensor[layers.length + 1];
        activated[0] = in.data;

        for (int i = 0; i < layers.length; i++) {
            output[i] = layers[i].generateOutput(weights[i], activated[i]);
            activated[i+1] = layers[i].activate(output[i]);
        }

        Tensor delta = activated[layers.length].sub(in.label);
        System.out.println(delta.mul(delta).sum());
        
        for (int i = layers.length-1; i >= 0; i--) {
            delta = layers[i].deriveDelta(activated[i+1], delta);
            gradient[i] = layers[i].gradient(activated[i], delta);

            if (i > 0)
                delta = layers[i].calculateBackprop(weights[i], delta);
        }
        return gradient;
    }

    public Tensor predict(Sample in) {
        Tensor activatedOut = in.data;
        for (int i = 0; i < layers.length; i++) {
            Tensor output = layers[i].generateOutput(weights[i], activatedOut);
            activatedOut = layers[i].activate(output);
        }
        return activatedOut;
    }

    public void updateWeight(Weight[] deltaWeight) {
        if (weights.length != deltaWeight.length)
            throw new IllegalArgumentException(String.format
                    ("Number of layers mismatch; current %d, deltaWeight %d", weights.length, deltaWeight.length));

        for (int i = 0 ; i < weights.length; i++) {
            if (weights[i] != null) {
                weightUpdates[i].w.muli(momentum);
                weightUpdates[i].w.subi(weights[i].w.mul(learningRate * decayLambda));
                if (deltaWeight[i] != null)
                    weightUpdates[i].w.subi(deltaWeight[i].w.mul(learningRate));
                
                weightUpdates[i].b.muli(momentum);
                if (deltaWeight[i] != null)
                    weightUpdates[i].b.subi(deltaWeight[i].b.mul(learningRate));

                weights[i].w.addi(weightUpdates[i].w);
                weights[i].b.addi(weightUpdates[i].b);
            }
        }
    }
}
