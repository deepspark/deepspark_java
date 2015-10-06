package org.acl.deepspark.nn.layers;

import org.acl.deepspark.data.Tensor;
import org.acl.deepspark.data.Weight;
import org.acl.deepspark.data.WeightFactory;
import org.acl.deepspark.data.WeightType;
import org.acl.deepspark.nn.conf.LayerConf;
import org.acl.deepspark.nn.functions.Activator;
import org.acl.deepspark.nn.functions.ActivatorFactory;
import org.acl.deepspark.nn.functions.ActivatorType;
import org.acl.deepspark.utils.ArrayUtils;

import java.io.Serializable;


// Fully Connected HiddenLayer
public class FullyConnectedLayer extends BaseLayer implements Serializable {
	private int 		dimOut;
	private boolean gpuAccel;
	private Activator 	activator;

	private static final long serialVersionUID = 2662945560065918864L;

	public FullyConnectedLayer(int[] inputShape, LayerConf conf, boolean gpuAccel) {
		super(inputShape);
		dimOut = (Integer) conf.get("num_output");
		activator = ActivatorFactory.get((ActivatorType) conf.get("activator"));
		this.gpuAccel = gpuAccel;

		System.out.println(String.format("dimOut: %d", dimOut));
		System.out.println(String.format("gpuAccel: %s", gpuAccel ? "true" : "false"));
	}

	@Override
	public Tensor generateOutput(Weight weight, Tensor input) {
		Tensor data = ArrayUtils.makeRowVector(input);
		return data.mmul(weight.w, gpuAccel).addi(weight.b);
	}

	@Override
	public Tensor deriveDelta(Tensor activated, Tensor error) {
		return error.mul(activator.derivative(activated));
	}

	@Override
	public Weight gradient(Tensor input,Tensor error) {
		Tensor data = ArrayUtils.makeColumnVector(input);
		return new Weight(data.mmul(error, gpuAccel), error);
	}

	@Override
	public Weight createWeight(LayerConf conf, int[] input) {
		WeightType typeW, typeB;
		float valueW, valueB;
		int dimIn = input[1]*input[2]*input[3];					// channels * rows * columns

		typeW = (WeightType) conf.get("weight_type");
		typeB = (WeightType) conf.get("bias_type");

		if (typeW == WeightType.XAVIER) {
			valueW = (float) Math.sqrt(2.0/dimIn);
		}  else {
			valueW = (conf.get("weight_value") == null) ?
					Weight.DEFAULT_VALUE : (Float) conf.get("weight_value");
		}

		if (typeB == WeightType.XAVIER) {
			valueB = (float) Math.sqrt(2.0/dimIn);
		} else {
			valueB = (conf.get("bias_value") == null) ?
					Weight.DEFAULT_VALUE : (Float) conf.get("bias_value");
		}

		if (typeW == null) typeW = Weight.DEFAULT_TYPE;
		if (typeB == null) typeB = Weight.DEFAULT_TYPE;

		return new Weight  (WeightFactory.create(typeW, valueW, dimIn, dimOut),
							WeightFactory.create(typeB, valueB, dimOut));
	}

	@Override
	public Tensor activate(Tensor output) {
		return activator.output(output);
	}

	@Override
	public int[] calcOutputShape() {
		return new int[] { getDimIn()[0], 1, 1, dimOut };
	}

	@Override
	public Tensor calculateBackprop(Weight weight, Tensor delta) {
		Tensor data = weight.w.mmul(delta.transpose(), gpuAccel);
		return data.reshape(getDimIn());
	}
}
