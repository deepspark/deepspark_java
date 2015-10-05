package org.acl.deepspark.nn.layers;

import org.acl.deepspark.data.Tensor;
import org.acl.deepspark.data.Weight;
import org.acl.deepspark.nn.conf.LayerConf;
import org.acl.deepspark.nn.functions.ActivatorType;

public class PoolingLayerTest {
	public static void main(String[] args) {

		float[] data = new float[] {1, 2, 3, 123, 2, 6 ,7, 8, 9, 10, 11, 12,
									13, 14, -3, 25, 17, -2, 19, 1, 21, 5, 23, 24,
									12, 26, 15, 28, 39, 30, 0, 5, 14, 1, 35, 3,
									0, 21, 39, 2, 1, 3, 33, 44, 26, 0, 3, 14};

		int[] dimIn = new int[] {3, 4, 4};
		Tensor input = Tensor.create(data, dimIn);

		LayerConf layerConf = new LayerConf(LayerType.POOLING);
		layerConf.set("poolRow", 2);
		layerConf.set("poolCol", 2);
		layerConf.set("stride", 1);
		layerConf.set("activator", ActivatorType.NONE);

		PoolingLayer poolingLayer = new PoolingLayer(dimIn, layerConf);
		Weight weight = poolingLayer.createWeight(layerConf, dimIn);
		Tensor output = poolingLayer.generateOutput(weight, input);


		System.out.println(String.format("input dim : (%d, %d, %d, %d)", input.shape()[0], input.shape()[1], input.shape()[2], input.shape()[3]));
		System.out.println(input);

		System.out.println(String.format("output dim : (%d, %d, %d, %d)", output.shape()[0], output.shape()[1], output.shape()[2], output.shape()[3]));
		System.out.println(output);

		Tensor delta = output.dup();
		Tensor propDelta = poolingLayer.calculateBackprop(weight, delta);

		System.out.println(String.format("delta dim : (%d, %d, %d, %d)", propDelta.shape()[0], propDelta.shape()[1], propDelta.shape()[2], propDelta.shape()[3]));
		System.out.println(propDelta);


		/** feedforward test complete **/
		/** back propagation test complete **/
	}
	
}
