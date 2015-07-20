package org.acl.deepspark.nn.layers;

import java.io.Serializable;

import org.acl.deepspark.data.Weight;
import org.acl.deepspark.nn.conf.LayerConf;
import org.acl.deepspark.nn.functions.Activator;
import org.acl.deepspark.nn.functions.ActivatorFactory;
import org.acl.deepspark.nn.functions.ActivatorType;
import org.acl.deepspark.utils.ArrayUtils;
import org.nd4j.linalg.api.ndarray.INDArray;



// Fully Connected HiddenLayer
public class FullyConnectedLayer implements Layer, Serializable {
	private Activator activator;
	public FullyConnectedLayer(ActivatorType t) {
		activator = ActivatorFactory.getActivator(t);
	}

	/**
	 * 
	 */
	private static final long serialVersionUID = 2662945560065918864L;

	@Override
	public INDArray generateOutput(Weight weight, INDArray input) {
		INDArray data = ArrayUtils.makeColumnVector(input);
		INDArray output = weight.w.mul(data).addi(weight.b);
		return activator.output(output);
	}

	@Override
	public INDArray deriveDelta(Weight w, INDArray error, INDArray output) {
		return w.w.transpose().mmuli(error).muli(activator.derivative(output));
	}

	@Override
	public Weight gradient(INDArray error, INDArray input) {
		Weight w = new Weight();
		w.w = error.mul(input);
		w.b = error;
		return w;
	}

	@Override
	public Weight createWeight(LayerConf conf, int[] input) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public INDArray activate(INDArray output) {
		return activator.output(output);
	}
	
}
