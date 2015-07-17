package org.acl.deepspark.nn.layers;

import org.acl.deepspark.data.Weight;
import org.acl.deepspark.nn.conf.LayerConf;
import org.acl.deepspark.utils.ArrayUtils;
import org.nd4j.linalg.api.activation.ActivationFunction;
import org.nd4j.linalg.api.ndarray.INDArray;



// Fully Connected HiddenLayer
public class FullyConnectedLayer implements Layer {
	private ActivationFunction activator;
	public FullyConnectedLayer(ActivationFunction f) {
		activator = f;
	}

	/**
	 * 
	 */
	private static final long serialVersionUID = 2662945560065918864L;

	@Override
	public INDArray generateOutput(Weight weight, INDArray input) {
		INDArray data = ArrayUtils.makeColumnVector(input);
		INDArray output = weight.w.mul(data).addi(weight.b);
		return activator.apply(output);
	}

	@Override
	public INDArray deriveDelta(Weight w, INDArray error, INDArray output) {
		activator.applyDerivative(input)
		return null;
	}

	@Override
	public INDArray gradient(INDArray delta, INDArray input) {
		
		return delta.mul(input);
	}

	@Override
	public INDArray createWeight(LayerConf conf, int[] input) {
		// TODO Auto-generated method stub
		return null;
	}
	
}
