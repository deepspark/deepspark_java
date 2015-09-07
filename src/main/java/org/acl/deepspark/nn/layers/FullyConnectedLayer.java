package org.acl.deepspark.nn.layers;

import java.io.Serializable;

import org.acl.deepspark.data.Tensor;
import org.acl.deepspark.data.Weight;
import org.acl.deepspark.nn.conf.LayerConf;
import org.acl.deepspark.nn.functions.Activator;
import org.acl.deepspark.nn.functions.ActivatorFactory;
import org.acl.deepspark.nn.functions.ActivatorType;
import org.acl.deepspark.utils.ArrayUtils;


// Fully Connected HiddenLayer
public class FullyConnectedLayer extends BaseLayer implements Serializable {
	private int 		numOut;
	private Activator 	activator;
	private static final long serialVersionUID = 2662945560065918864L;

	public FullyConnectedLayer(int[] inputShape, LayerConf conf) {
		super(inputShape);
		numOut = (Integer) conf.get("numNodes");
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
		Tensor data = ArrayUtils.makeRowVector(input);
		Weight w = new Weight();
		w.w = data.transpose().mmul(error);
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
		w.w = Tensor.randn(dimIn, dimOut).mul(0.1);//.mul(Math.sqrt(2.0/dimIn));
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
/*
	@Override
	public INDArray generateOutputBatch(Weight weight, INDArray input) {
		int numSample = input.shape()[0];
		int length = input.slice(0).length();
		
		INDArray data = Nd4j.create(numSample, length);
		INDArray bias = Nd4j.create(numSample, weight.b.length());
		for(int i = 0; i < numSample; i++) {
			data.putSlice(i, input.slice(i).reshape(length));
			bias.putSlice(i, weight.b.linearView());
		}
		data.transposei();
		bias.transposei();
		
		return weight.w.mmul(data).addi(bias);
	}

	@Override
	public Weight gradientBatch(INDArray input, INDArray error) {
		int numSample = input.shape()[0];
		int inputLength = input.slice(0).length();
		int errorLength = error.slice(0).length();
		
		INDArray data = Nd4j.create(numSample, inputLength);
		INDArray delta = Nd4j.create(numSample, errorLength);
		
		/*
		 * delta = sample x output
		 * data = sample x feature
		 *
		for(int i = 0; i < numSample; i++) {
			data.putSlice(i, input.slice(i).reshape(inputLength));
			delta.putSlice(i, error.slice(i).reshape(errorLength));
		}
		data.transposei();
		
		/*
		 * data = feature x sample
		 * w = output x feature
		 *
 		
		Weight w = new Weight();
		w.w = data.mul(delta).transposei().div(numSample);  // output x feature
		w.b = delta.sum(0).div(numSample); 
		return w;
	}

	@Override
	public INDArray calculateBackpropBatch(Weight weight, INDArray error) {
		int numSample = error.shape()[0];
		int errorLength = error.slice(0).length();
		
		// delta = output x sample
		INDArray delta = Nd4j.create(numSample, errorLength);
		for(int i = 0; i < numSample; i++)
			delta.putSlice(i, error.slice(i).reshape(errorLength));
		delta.transposei();
		
		// data = sample x feature
		INDArray data = weight.w.transpose().mmul(delta).transposei();
		
		int dim[] = new int[getInputShape().length + 1];
		dim[0] = numSample;
		System.arraycopy(getInputShape(), 0, dim, 1, dim.length - 1);
		
		// data = sample x (input dim)
		return data.reshape(dim);
	}
*/
}
