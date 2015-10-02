package org.acl.deepspark.nn.layers;

import org.acl.deepspark.data.Tensor;
import org.acl.deepspark.data.Weight;
import org.acl.deepspark.nn.conf.LayerConf;
import org.acl.deepspark.nn.functions.Activator;
import org.acl.deepspark.nn.functions.ActivatorFactory;
import org.acl.deepspark.nn.functions.ActivatorType;
import org.acl.deepspark.utils.ArrayUtils;

import java.io.Serializable;


// Fully Connected HiddenLayer
public class FullyConnectedLayer extends BaseLayer implements Serializable {
	private int 		numOut;
	private Activator 	activator;

	private float std;
	private static final long serialVersionUID = 2662945560065918864L;

	public FullyConnectedLayer(int[] inputShape, LayerConf conf) {
		super(inputShape);
		numOut = (Integer) conf.get("numNodes");
		std = (Float) conf.get("std");
		activator = ActivatorFactory.get((ActivatorType) conf.get("activator"));
	}

	@Override
	public Tensor generateOutput(Weight weight, Tensor input) {

		Tensor data = ArrayUtils.makeRowVector(input);
		return data.mmul(weight.w).addi(weight.b);
	}

	@Override
	public Tensor deriveDelta(Tensor activated, Tensor error) {
		return error.mul(activator.derivative(activated));
	}

	@Override
	public Weight gradient(Tensor input,Tensor error) {
		Tensor data = ArrayUtils.makeColumnVector(input);
		Weight w = new Weight();
		w.w = data.mmul(error);
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

		w.w = Tensor.randn(dimIn, dimOut).muli(std);
		//w.w = Tensor.randn(dimIn, dimOut).mul((float) Math.sqrt(2.0/dimIn));
		w.b = Tensor.zeros(dimOut);
		return w;
	}

	@Override
	public Tensor activate(Tensor output) {
		return activator.output(output);
	}

	@Override
	public int[] calculateOutputDimension() {
		return new int[] { numOut };
	}

	@Override
	public Tensor calculateBackprop(Weight weight, Tensor delta) {
		Tensor data = weight.w.mmul(delta.transpose());
		return data.reshape(getInputShape());
	}
}
