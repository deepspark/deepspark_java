package org.acl.deepspark.nn.layers;

import org.acl.deepspark.data.Weight;
import org.acl.deepspark.nn.conf.LayerConf;
import org.acl.deepspark.nn.functions.ActivatorType;
import org.acl.deepspark.nn.layers.LayerType;
import org.acl.deepspark.nn.layers.PoolingLayer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class PoolingLayerTest {
	public static void main(String[] args) {

		double[] data = new double[] {1, 2, 3, 123, 2, 6 ,7, 8, 9, 10, 11, 12,
									13, 14, 15, 25, 17, 18, 19, 20, 21, 5, 23, 24,
									25, 26, 15, 28, 39, 30, 31, 32, 14, 34, 35, 36,
									0, 21, 39, 2, 1, 3, 33, 44, 26, 0, 3, 14};

		int[] dimIn = new int[] {3, 4, 4};
		INDArray input = Nd4j.create(data, dimIn);

		LayerConf layerConf = new LayerConf(LayerType.POOLING);
		layerConf.set("poolRow", 2);
		layerConf.set("poolCol", 2);
		layerConf.set("activator", ActivatorType.SIGMOID);

		PoolingLayer poolingLayer = new PoolingLayer(dimIn, layerConf);
		Weight weight = poolingLayer.createWeight(layerConf, dimIn);
		int[] dimOut = poolingLayer.calculateOutputDimension(layerConf, dimIn);
		INDArray output = poolingLayer.generateOutput(weight, input);

		int channelIdx = 0;

		System.out.println(String.format("input dim : (%d, %d, %d)", input.size(0), input.size(1), input.size(2)));
		System.out.println(input.slice(channelIdx));

		System.out.println(String.format("output dim : (%d, %d, %d)", output.size(0), output.size(1), output.size(2)));
		System.out.println(output.slice(channelIdx));

		INDArray delta = Nd4j.ones(output.shape());
		INDArray propDelta = poolingLayer.calculateBackprop(null, delta);

		System.out.println(String.format("delta dim : (%d, %d, %d)", propDelta.size(0), propDelta.size(1), propDelta.size(2)));
		System.out.println(propDelta.slice(channelIdx));
	}
	
}
