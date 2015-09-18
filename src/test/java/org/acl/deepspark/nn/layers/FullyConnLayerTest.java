package org.acl.deepspark.nn.layers;


import org.acl.deepspark.data.Tensor;
import org.acl.deepspark.data.Weight;
import org.acl.deepspark.nn.conf.LayerConf;
import org.acl.deepspark.nn.functions.ActivatorType;

public class FullyConnLayerTest {
	public static void main(String[] args) {
		float[] data = new float[] {1, 2, 1, 5, 1, 4, 8, 3, 2, 7, 9, 3,
				5, 8, 3, 4, 1, 12, 23, 34, 1, 4, 2, 1,
				4, 5, 23, 2, 1, 5, 7, 23, 1, 2, 4, 7};

		int outUnit = 10;

		int[] dimIn = new int[] {4, 3, 3};
		Tensor input = Tensor.create(data, dimIn);
		System.out.println("input");
		System.out.println(input);
		System.out.println(input.reshape(data.length));

		LayerConf layerConf = new LayerConf(LayerType.FULLYCONN);
		layerConf.set("numNodes", outUnit);
		layerConf.set("activator", ActivatorType.RECTIFIED_LINEAR);

		FullyConnectedLayer fullyConnectedLayer = new FullyConnectedLayer(dimIn, layerConf);
		Weight weight = fullyConnectedLayer.createWeight(layerConf, dimIn);
		System.out.println("weight");
		System.out.println(weight);

		Tensor output = fullyConnectedLayer.generateOutput(weight, input);
		System.out.println("output");
		System.out.println(output);

		System.out.println(String.format("input dim: (%d, %d, %d, %d)", input.shape()[0], input.shape()[1], input.shape()[2], input.shape()[3]));
		System.out.println(String.format("weight dim: (%d, %d, %d, %d)", weight.w.shape()[0], weight.w.shape()[1], weight.w.shape()[2], weight.w.shape()[3]));
		System.out.println(String.format("bias dim: (%d, %d, %d, %d)", weight.b.shape()[0], weight.b.shape()[1], weight.b.shape()[2], weight.b.shape()[3]));
		System.out.println(String.format("output dim: (%d, %d, %d, %d)", output.shape()[0], output.shape()[1], output.shape()[2], output.shape()[3]));

		Tensor propDelta = fullyConnectedLayer.calculateBackprop(weight, output);
		System.out.println(String.format("delta dim: (%d, %d, %d, %d)", propDelta.shape()[0], propDelta.shape()[1], propDelta.shape()[2], propDelta.shape()[3]));

		/** feedforward test complete **/
	}
}
