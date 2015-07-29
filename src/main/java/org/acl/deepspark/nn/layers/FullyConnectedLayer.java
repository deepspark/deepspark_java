package org.acl.deepspark.nn.layers;

import java.io.Serializable;

import org.acl.deepspark.data.Weight;
import org.acl.deepspark.nn.conf.LayerConf;
import org.acl.deepspark.utils.ArrayUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;



// Fully Connected HiddenLayer
public class FullyConnectedLayer extends BaseLayer implements Serializable {

	public FullyConnectedLayer(int[] inputShape, LayerConf conf) {
		super(inputShape, conf);
	}

	/**
	 * 
	 */
	private static final long serialVersionUID = 2662945560065918864L;
	// complete// complete
	@Override
	public INDArray generateOutput(Weight weight, INDArray input) {
		INDArray data = ArrayUtils.makeColumnVector(input);
		System.out.println("fullyconn output:" + weight.w.mmul(data).addi(weight.b));
		return weight.w.mmul(data).addi(weight.b);
	}

	// complete
	@Override
	public INDArray deriveDelta(INDArray output, INDArray error) {
		System.out.println("fullyconn error:" + error);
		System.out.println("fullyconn derivative:" + error.mul(activator.derivative(output)));
		return error.mul(activator.derivative(output));
	}

	// complete
	@Override
	public Weight gradient(INDArray input,INDArray error) {
		INDArray data = ArrayUtils.makeColumnVector(input);
		Weight w = new Weight();
		w.w = error.mmul(data.transpose());
		w.b = error;
		System.out.println("fullyconn gradient:" + w.toString());
		return w;
	}

	// complete
	@Override
	public Weight createWeight(LayerConf conf, int[] input) {
		int dimOut = (Integer) conf.get("numNodes");
		int dimIn = 1;
		for(int i =0; i < input.length; i++)
			dimIn *= input[i];

		Weight w= new Weight();
		w.w = Nd4j.randn(dimOut, dimIn);
		w.b = Nd4j.zeros(dimOut, 1).mul(0.01);
		return w;
	}

	// complete
	@Override
	public INDArray activate(INDArray output) {
		System.out.println("fullyconn activate:" + activator.output(output));
		return activator.output(output);
	}

	// complete
	@Override
	public int[] calculateOutputDimension(LayerConf conf, int[] input) {
		int[] ret = new int[2];
		ret[0] = (Integer) conf.get("numNodes");
		ret[1] = 1;
		return ret;
	}
	// complete
	@Override
	public INDArray calculateBackprop(Weight weight, INDArray delta) {
		INDArray data = weight.w.transpose().mmul(delta);
		System.out.println("fullyconn delta:" + delta);
		System.out.println("fullyconn backprop:" + data);
		return data.reshape(getInputShape());
	}
}
