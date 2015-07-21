package org.acl.deepspark.nn.layers;

import java.io.Serializable;

import org.acl.deepspark.data.Weight;
import org.acl.deepspark.nn.conf.LayerConf;
import org.acl.deepspark.nn.functions.Activator;
import org.acl.deepspark.nn.functions.ActivatorFactory;
import org.acl.deepspark.nn.functions.ActivatorType;
import org.acl.deepspark.utils.ArrayUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;



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
		int dimOut = (Integer) conf.get("numNodes");
		int dimIn = 1; 
		for(int i =0; i < input.length; i++)
			dimIn *= input[i];
		
		Weight w= new Weight();
		w.w = Nd4j.randn(dimOut, dimIn);
		w.b = Nd4j.ones(dimOut).mul(0.01);
		return w;
	}

	@Override
	public INDArray activate(INDArray output) {
		return activator.output(output);
	}

	@Override
	public int[] calculateOutputDimension(LayerConf conf, int[] input) {
		int[] ret = new int[1];
		ret[0] = (Integer) conf.get("numNodes");
		return ret;
	}
}
