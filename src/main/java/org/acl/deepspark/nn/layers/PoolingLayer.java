package org.acl.deepspark.nn.layers;

import java.io.Serializable;

import org.acl.deepspark.data.Weight;
import org.acl.deepspark.nn.conf.LayerConf;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class PoolingLayer extends BaseLayer implements Serializable, Layer {
	/**
	 * 
	 */
	private static final long serialVersionUID = -4318643106939173007L;

	private int poolRow;
	private int poolCol;
	private INDArray maskArray;

	public PoolingLayer(int[] inputShape, LayerConf conf) {
		super(inputShape, conf);
		this.poolRow = (Integer) conf.get("poolRow");
		this.poolCol = (Integer) conf.get("poolCol");
	}

	@Override
	public Weight createWeight(LayerConf conf, int[] input) {
		return null;
	}

	@Override
	public int[] calculateOutputDimension(LayerConf conf, int[] input) {
		return new int[] {input[0], input[1]/poolRow, input[2]/poolCol};
	}

	@Override
	public INDArray generateOutput(Weight weight, INDArray input) {
		int numChannel = input.size(0);
		int rows = input.size(1);
		int cols = input.size(2);

		double value;
		double outValue;

		INDArray output = Nd4j.create(numChannel, (rows / poolRow), (cols / poolCol));
		maskArray = Nd4j.create(output.shape());
		for (int ch = 0; ch < numChannel; ch++) {
			for (int r = 0; r < rows; r++) {
				int or = r / poolRow;
				for (int c = 0; c < cols; c++) {
					int oc = c / poolCol;
					value = input.getDouble(ch, r, c);
					outValue = output.getDouble(ch, or, oc);
					if (value > outValue) {
						output.putScalar(new int[]{ch, or, oc}, value);
						maskArray.putScalar(new int[]{ch, or, oc}, input.slice(ch).index(r, c));
					}
				}
			}
		}
		return output;
	}

	@Override
	public INDArray activate(INDArray output) {
		return output;
	}

	@Override
	public INDArray deriveDelta(INDArray output, INDArray error) {
		return error;
	}

	@Override
	public Weight gradient(INDArray input, INDArray error) {
		return null;
	}

	@Override
	public INDArray calculateBackprop(Weight weight, INDArray error) {
		INDArray propDelta = Nd4j.create(getInputShape());
		int numChannel = propDelta.size(0);
		int rows = error.size(1);
		int cols = error.size(2);

		for (int ch = 0; ch < numChannel; ch++) {
			for (int or = 0; or < rows; or++) {
				for (int oc = 0; oc < cols; oc++) {
					propDelta.putScalar(maskArray.getInt(ch, or, oc),
										error.getDouble(ch, or, oc));
				}
			}
		}
		return propDelta;
	}

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
}
