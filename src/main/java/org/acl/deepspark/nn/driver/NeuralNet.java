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

/**
 * Created by Jaehong on 2015-07-16.
 */
public class NeuralNet {


    private ArrayList<Layer> layers;
    private INDArray[] weights;
    private INDArray activationOut;
    private DeltaWeight deltaWeights;

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
        ArrayList<LayerConf> arr = conf.getLayerList();
        int[] dimIn = conf.getDimIn();

        layers = new ArrayList<>();
        weights = new INDArray[arr.size()];

        buildNetwork(arr, dimIn);
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
                layers.add(layer);
                weights[i] = layer.createWeight(layerConf, dimIn);
            }
        }
    }


    public INDArray feedForward(Sample in) {
        return null;
    }


    public void backPropagate(INDArray error) {

    }


}
