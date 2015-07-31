package org.acl.deepspark.nn.layers;
import org.acl.deepspark.data.Weight;
import org.acl.deepspark.nn.conf.LayerConf;
import org.acl.deepspark.nn.functions.ActivatorType;
import org.acl.deepspark.nn.layers.ConvolutionLayer;
import org.acl.deepspark.nn.layers.LayerType;
import org.acl.deepspark.utils.ArrayUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;


public class ConvolutionLayerTest {
	
	// FeedForward Test
	public static void main(String[] args) {

		double[] data = new double[] {1, 2, 1, 5, 1, 4, 8, 3, 2, 7, 9, 3,
									 5, 8, 3, 4, 1, 12, 23, 34, 1, 4, 2, 1,
									 4, 5, 23, 2, 1, 5, 7, 23, 1, 2, 4, 7};

		int[] dimIn = new int[] {4, 3, 3};
		INDArray input = Nd4j.create(data, dimIn);
		LayerConf layerConf = new LayerConf(LayerType.CONVOLUTION);
		layerConf.set("numFilters", 5);
		layerConf.set("filterRow", 2);
		layerConf.set("filterCol", 2);
		layerConf.set("activator", ActivatorType.SIGMOID);

		ConvolutionLayer convLayer = new ConvolutionLayer(dimIn, layerConf);
		Weight weight = convLayer.createWeight(layerConf, dimIn);
		int[] dimOut = convLayer.calculateOutputDimension(layerConf, dimIn);

		INDArray output = convLayer.generateOutput(weight, input);

		System.out.println(String.format("input dim: (%d, %d, %d)", input.size(0), input.size(1), input.size(2)));
		System.out.println(String.format("weight dim: (%d, %d, %d, %d)", weight.w.size(0), weight.w.size(1), weight.w.size(2), weight.w.size(3)));
		System.out.println(String.format("bias dim: (%d, %d)", weight.b.size(0), weight.b.size(1)));
		System.out.println(String.format("output dim: (%d, %d, %d)", output.size(0), output.size(1), output.size(2)));

		INDArray propDelta = convLayer.calculateBackprop(weight, output);
		System.out.println(String.format("delta dim: (%d, %d, %d)", propDelta.size(0), propDelta.size(1), propDelta.size(2)));



		System.out.println(input);
		INDArray ret = ArrayUtils.makeColumnVector(input);

		System.out.println(ret.reshape(dimIn));


	}

		/** feedforward test complete **/


}
