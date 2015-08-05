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

	@Override
	public INDArray generateOutput(Weight weight, INDArray input) {
		INDArray data = ArrayUtils.makeColumnVector(input);
		return weight.w.mmul(data).addi(weight.b);
	}

	@Override
	public INDArray deriveDelta(INDArray output, INDArray error) {
		return error.mul(activator.derivative(output,true));
	}

	@Override
	public Weight gradient(INDArray input,INDArray error) {
		INDArray data = ArrayUtils.makeColumnVector(input);
		Weight w = new Weight();
		w.w = error.mmul(data.transpose());
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
		w.w = Nd4j.randn(dimOut, dimIn).mul(0.1);
		w.b = Nd4j.zeros(dimOut, 1);
		return w;
	}

	@Override
	public INDArray activate(INDArray output) {
		return activator.output(output);
	}

	@Override
	public int[] calculateOutputDimension(LayerConf conf, int[] input) {
		int[] ret = new int[2];
		ret[0] = (Integer) conf.get("numNodes");
		ret[1] = 1;
		return ret;
	}

	@Override
	public INDArray calculateBackprop(Weight weight, INDArray delta) {
		INDArray data = weight.w.transpose().mmul(delta);
		return data.reshape(getInputShape());
	}

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
		 */
		for(int i = 0; i < numSample; i++) {
			data.putSlice(i, input.slice(i).reshape(inputLength));
			delta.putSlice(i, error.slice(i).reshape(errorLength));
		}
		data.transposei();
		
		/*
		 * data = feature x sample
		 * w = output x feature
		 */
 		
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
}
