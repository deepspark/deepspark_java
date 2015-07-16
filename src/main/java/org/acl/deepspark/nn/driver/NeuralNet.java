package org.acl.deepspark.nn.driver;

import org.acl.deepspark.data.DeltaWeight;
import org.acl.deepspark.data.Sample;
import org.acl.deepspark.data.Weight;
import org.acl.deepspark.nn.conf.LayerConf;
import org.acl.deepspark.nn.conf.NeuralNetConf;
import org.acl.deepspark.nn.functions.Activator;
import org.acl.deepspark.nn.layers.BaseLayer;
import org.acl.deepspark.nn.layers.FullyConnLayer;
import org.acl.deepspark.nn.layers.Layer;
import org.acl.deepspark.nn.layers.cnn.ConvolutionLayer;
import org.acl.deepspark.nn.layers.cnn.PoolingLayer;
import org.jblas.DoubleMatrix;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.ListIterator;

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
        buildLayer(conf);
    }

    private void buildLayer(final NeuralNetConf conf) {
        layers = new ArrayList<>();
        ArrayList<LayerConf> arr = conf.getLayerList();

        int type;
        for (int i = 0 ; i< arr.size(); i++) {
            type = arr.get(i).getType();
            switch (type) {
                case LayerConf.CONVOLUTION:
                    layers.add(new ConvolutionLayer());
                    break;

                case LayerConf.POOLING:
                    layers.add(new PoolingLayer());
                    break;

                case LayerConf.FULLYCONN:
                    layers.add(new FullyConnLayer());
                    break;
            }
        }
        createWeight(arr);
    }

    private void createWeight(final ArrayList<LayerConf> arr) {
        
    }


    public INDArray feedForward(Sample in) {
        return null;
    }


    public void backPropagate(INDArray error) {

    }


}
