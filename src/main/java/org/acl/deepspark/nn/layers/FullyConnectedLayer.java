package org.acl.deepspark.nn.layers;

import org.acl.deepspark.nn.functions.Activator;
import org.acl.deepspark.utils.ArrayUtils;
import org.nd4j.linalg.api.ndarray.INDArray;



// Fully Connected HiddenLayer
public class FullyConnectedLayer extends BaseLayer{

	public FullyConnectedLayer(Activator f) {
		super(f);
	}

	/**
	 * 
	 */
	private static final long serialVersionUID = 2662945560065918864L;

	@Override
	public INDArray generateOutput(INDArray w, INDArray input) {
		INDArray data = ArrayUtils.makeColumnVector(input);
		
		return w.mmul(data);
	}

	@Override
	public INDArray deriveDelta(INDArray w, INDArray error, INDArray output) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public INDArray gradient(INDArray delta, INDArray input) {
		
		return delta.mul(input);
	}
	
	
}
