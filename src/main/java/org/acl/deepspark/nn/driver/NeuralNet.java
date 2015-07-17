package org.acl.deepspark.nn.driver;

import org.acl.deepspark.data.DeltaWeight;
import org.acl.deepspark.data.Sample;
import org.acl.deepspark.nn.conf.LayerConf;
import org.acl.deepspark.nn.conf.NeuralNetConf;
import org.acl.deepspark.nn.layers.FullyConnLayer;
import org.acl.deepspark.nn.layers.Layer;
import org.acl.deepspark.nn.layers.cnn.ConvolutionLayer;
import org.acl.deepspark.nn.layers.cnn.PoolingLayer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.Iterator;

/**
 * Created by Jaehong on 2015-07-16.
 */
public class NeuralNet {
    private Layer[] layers;
    private INDArray[] weights;
    private INDArray[] deltaWeights;
    private INDArray[] activationOut;

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
        weights = new INDArray[size];
        deltaWeights = new INDArray[size];
        activationOut = new INDArray[size+1];

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

    public void setWeights(INDArray[] weights) {
        this.weights = weights;
    }

    public INDArray[] getWeights() {
        return weights;
    }

    public void setDeltaWeights(INDArray[] deltaWeights) {
        this.deltaWeights = deltaWeights;
    }

    public INDArray[] getDeltaWeights() {
        return deltaWeights;
    }

    public INDArray feedForward(Sample in) {
        activationOut[0] = in.getFeature();
        for (int i = 0; i < layers.length; i++) {
            activationOut[i+1] = layers[i].generateOutput(weights[i], activationOut[i]);
        }
        return activationOut[layers.length];
    }

    public void backPropagate(INDArray error) {
        for (int i = layers.length-1; i >= 0; i--) {
            deltaWeights[i] = layers[i].gradient(activationOut[i], error);
            error = layers[i].deriveDelta(weights[i], error);
        }
    }


    public void updateWeight() {
        for (int i = 0 ; i < weights.length; i++) {
            // TODO update weight w/ lr and acummulated deltaWeights
        }
    }
}
