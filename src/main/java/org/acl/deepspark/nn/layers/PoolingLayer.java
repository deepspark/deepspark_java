package org.acl.deepspark.nn.layers;

import org.acl.deepspark.data.Tensor;
import org.acl.deepspark.data.Weight;
import org.acl.deepspark.nn.conf.LayerConf;

import java.io.Serializable;

public class PoolingLayer extends BaseLayer implements Serializable, Layer {
	/**
	 * 
	 */
	private static final long serialVersionUID = -4318643106939173007L;

	private int poolRow;
	private int poolCol;
	private int strides = 1;

	public PoolingLayer(int[] inputShape, LayerConf conf) {
		super(inputShape);
		this.poolRow = (Integer) conf.get("poolRow");
		this.poolCol = (Integer) conf.get("poolCol");
		this.strides = poolRow;
	}

	@Override
	public Weight createWeight(LayerConf conf, int[] input) {
		Weight weight = new Weight();
		weight.w = Tensor.zeros(calculateOutputDimension());
		weight.b = Tensor.zeros(1);
		return weight;
	}

	@Override
	public int[] calculateOutputDimension() {
		int[] dimIn = getInputShape();
		return new int[] {dimIn[0], (dimIn[1]-poolRow)/strides + 1, (dimIn[2]-poolCol)/strides + 1};
	}

	@Override
	public Tensor generateOutput(Weight weight, Tensor input) {
		int[] dimIn = input.shape();

		double value;
		double outValue;

		Tensor output = Tensor.ones(calculateOutputDimension()).mul(Double.MAX_VALUE * -1);
		for (int k = 0; k < dimIn[0]; k++) {
			for (int ch = 0; ch < dimIn[1]; ch++) {
				for (int r = 0; r < dimIn[2]; r++) {
					int or = r / poolRow;
					for (int c = 0; c < dimIn[3]; c++) {
						int oc = c / poolCol;

						value = input.slice(k, ch).get(r, c);
						outValue = output.slice(k, ch).get(or, oc);
						if (value > outValue) {
							output.slice(k, ch).put(output.slice(k, ch).index(or, oc), value);
							weight.w.slice(k, ch).put(weight.w.slice(k, ch).index(or, oc), input.slice(k, ch).index(r, c));
						}
					}
				}
			}
		}
		return output;
	}

	@Override
	public Tensor activate(Tensor output) {
		return output;
	}

	@Override
	public Tensor deriveDelta(Tensor output, Tensor error) {
		return error;
	}

	@Override
	public Weight gradient(Tensor input, Tensor error) {
		return null;
	}

	@Override
	public Tensor calculateBackprop(Weight weight, Tensor error) {
		Tensor propDelta = Tensor.zeros(getInputShape());
		int[] dimOut = calculateOutputDimension();

		for (int ch = 0; ch < dimOut[0]; ch++) {
			for (int or = 0; or < dimOut[1]; or++) {
				for (int oc = 0; oc < dimOut[2]; oc++) {
					propDelta.slice(0, ch).put((int) weight.w.slice(0, ch).get(or, oc),
							error.slice(0, ch).get(or, oc));
				}
			}
		}
		return propDelta;
	}
/*
	@Override
	public INDArray generateOutputBatch(Weight weight, INDArray input) {
		// TODO Auto-generated method stub
		int numSample= input.size(0);
		int numChannel = input.size(1);
		int rows = input.size(2);
		int cols = input.size(3);

		INDArray output = Nd4j.create(numSample, numChannel, (rows / poolRow), (cols / poolCol));
		for(int i =0; i < numSample; i++)
			output.putSlice(i, generateOutput(weight, input.slice(0)));

		return output;
	}

	@Override
	public Weight gradientBatch(INDArray input, INDArray error) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public INDArray calculateBackpropBatch(Weight weight, INDArray error) {
		int[] dim = new int[getInputShape().length +1];
		dim[0] = error.shape()[0];
		System.arraycopy(getInputShape(), 0, dim, 1, getInputShape().length);
		
		INDArray propDelta = Nd4j.create(dim);
		int numSample = dim[0];
		
		for(int i = 0; i < numSample; i++)
			propDelta.putSlice(i, calculateBackprop(weight, error.slice(0)));
			
		return propDelta;
	}
*/
}
