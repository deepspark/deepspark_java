package org.acl.deepspark.nn.layers;


import org.acl.deepspark.data.Weight;
import org.acl.deepspark.nn.conf.LayerConf;
import org.acl.deepspark.nn.functions.ActivatorType;
import org.acl.deepspark.nn.layers.FullyConnectedLayer;
import org.acl.deepspark.nn.layers.LayerType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class FullyConnLayerTest {
	public static void main(String[] args) {
		double[] data = new double[] {1, 2, 1, 5, 1, 4, 8, 3, 2, 7, 9, 3,
				5, 8, 3, 4, 1, 12, 23, 34, 1, 4, 2, 1,
				4, 5, 23, 2, 1, 5, 7, 23, 1, 2, 4, 7};

		int outUnit = 120;

		int[] dimIn = new int[] {4, 3, 3};
		INDArray input = Nd4j.create(data, dimIn);
		LayerConf layerConf = new LayerConf(LayerType.FULLYCONN);
		layerConf.set("numNodes", outUnit);
		layerConf.set("activator", ActivatorType.SIGMOID);


		FullyConnectedLayer fullyConnectedLayer = new FullyConnectedLayer(dimIn, layerConf);
		Weight weight = fullyConnectedLayer.createWeight(layerConf, dimIn);

		INDArray output = fullyConnectedLayer.generateOutput(weight, input);

		System.out.println(String.format("input dim: (%d, %d, %d)", input.size(0), input.size(1), input.size(2)));
		System.out.println(String.format("weight dim: (%d, %d)", weight.w.size(0), weight.w.size(1)));
		System.out.println(String.format("bias dim: (%d, %d)", weight.b.size(0), weight.b.size(1)));
		System.out.println(String.format("output dim: (%d, %d)", output.size(0), output.size(1)));

		INDArray propDelta = fullyConnectedLayer.calculateBackprop(weight, output);
		System.out.println(String.format("delta dim: (%d, %d, %d)", propDelta.size(0), propDelta.size(1), propDelta.size(2)));
	}
}
