package org.acl.deepspark.nn.driver;

import org.acl.deepspark.data.Accumulator;
import org.acl.deepspark.data.Sample;
import org.acl.deepspark.data.Weight;
import org.acl.deepspark.nn.conf.LayerConf;
import org.acl.deepspark.nn.conf.NeuralNetConf;
import org.acl.deepspark.nn.layers.FullyConnLayer;
import org.acl.deepspark.nn.layers.Layer;
import org.acl.deepspark.nn.layers.cnn.ConvolutionLayer;
import org.acl.deepspark.nn.layers.cnn.PoolingLayer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;

/**
 * Created by Jaehong on 2015-07-16.
 */
public class NeuralNet {
    private Layer[]     layers;
    private Weight[]  weights;

    double learningRate;
    double decayLambda;
    double momentum;
    double dropOutRate;

    public NeuralNet(final NeuralNetConf conf) {
        learningRate = conf.getParams().get("learningRate");
        decayLambda = conf.getParams().get("decayLambda");
        momentum = conf.getParams().get("momentum");
        dropOutRate = conf.getParams().get("dropOutRate");
        initNetwork(conf);
    }

    public void initNetwork(final NeuralNetConf conf) {
        int size = conf.getLayerList().size();
        layers = new Layer[size];
        weights = new Weight[size];
        buildNetwork(conf.getLayerList(), conf.getDimIn());
    }

    private void buildNetwork(ArrayList<LayerConf> arr, int[] dimIn) {
        for (int i = 0 ; i< arr.size(); i++) {
            LayerConf layerConf = arr.get(i);
            int activator = layerConf.getActivator();
            Layer layer = null;

            switch (layerConf.getType()) {
                case LayerConf.CONVOLUTION:
                    layer = new ConvolutionLayer(activator);
                    break;
                case LayerConf.POOLING:
                    layer = new PoolingLayer(activator);
                    break;
                case LayerConf.FULLYCONN:
                    layer = new FullyConnLayer(activator);
                    break;
            }
            if (layer != null) {
                layers[i] = layer;
                weights[i] = layers[i].createWeight(layerConf, dimIn);
            }
        }
    }

    public void setWeights(Weight[] weights) {
        this.weights = weights;
    }

    public Weight[] getWeights() {
        return weights;
    }

    public INDArray feedForward(Sample in) {
        INDArray activationOut = in.getFeature();
        for (int i = 0; i < layers.length; i++) {
//            preActivation[i + 1] = layers[i].generateOutput(weights[i], activationOut);
//            activationOut = layers[i].activate(preActivation[i + 1]);
            activationOut = layers[i].generateOutput(weights[i], activationOut);
        }
        return activationOut;
    }

    public Weight[] train(Sample in, INDArray label) {
        Weight[] gradient = new Weight[layers.length];
        INDArray[] preActivation = new INDArray[layers.length];
        INDArray[] postActivation = new INDArray[layers.length + 1];
        postActivation[0] = in.getFeature();

        for (int i = 0; i < layers.length; i++) {
            preActivation[i] = layers[i].generateOutput(weights[i], postActivation[i]);
            postActivation[i+1] = layers[i].activate(preActivation[i]);
        }

        INDArray error = postActivation[layers.length].sub(label);
        for (int i = layers.length-1; i >= 0; i--) {
            gradient[i] = layers[i].gradient(preActivation[i], postActivation[i], error);
            error = layers[i].deriveDelta(weights[i], preActivation[i], error);
        }
        return gradient;
    }
    /** move to NeuralNetRunner **/
//
//    public void calcDeltaWeight() {
//        INDArray[] arr = gradients.getAverage();
//        for (int i = 0 ; i < layers.length; i++) {
//            if (deltaWeights[i] == null)
//                deltaWeights[i] = weights[i].mul(decayLambda).add(arr[i]).mul(-1 * learningRate);
//            else {
//                INDArray delta = weights[i].mul(decayLambda).add(arr[i]).mul(-1 * learningRate);
//                deltaWeights[i].addi(deltaWeights[i].mul(momentum).add(delta));
//            }
//        }
//    }

    public void updateWeight(Weight deltaWeight) {
        for (int i = 0 ; i < weights.length; i++) {

        }
    }
}
