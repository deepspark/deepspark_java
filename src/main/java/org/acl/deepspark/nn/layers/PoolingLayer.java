package org.acl.deepspark.nn.layers;

import org.acl.deepspark.data.Tensor;
import org.acl.deepspark.data.Weight;
import org.acl.deepspark.nn.conf.LayerConf;
import org.jblas.DoubleMatrix;
import org.jblas.FloatMatrix;
import org.jblas.ranges.RangeUtils;

import java.io.Serializable;

public class PoolingLayer extends BaseLayer implements Serializable, Layer {
	/**
	 * 
	 */
	private static final long serialVersionUID = -4318643106939173007L;

	private int poolRow;
	private int poolCol;
	private int stride;

	public PoolingLayer(int[] inputShape, LayerConf conf) {
		super(inputShape);
		this.poolRow = (Integer) conf.get("poolRow");
		this.poolCol = (Integer) conf.get("poolCol");
		this.stride = (Integer) conf.get("stride");
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
		return new int[] {dimIn[0], (dimIn[1]-poolRow)/stride+1, (dimIn[2]-poolCol)/stride+1};
	}

	@Override
	public Tensor generateOutput(Weight weight, Tensor input) {

		int[] dimIn = getInputShape();
		int rowKernels = calculateOutputDimension()[1];
		int colKernels = calculateOutputDimension()[2];

		FloatMatrix subMat;
		Tensor poolOut = Tensor.zeros(calculateOutputDimension());

		for (int ch = 0; ch < dimIn[0]; ch++) {
			for (int c = 0; c < colKernels; c++) {
				for (int r = 0; r < rowKernels; r++) {
					subMat = input.slice(0, ch).get(RangeUtils.interval(r*stride, r*stride+poolRow), RangeUtils.interval(c*stride, c*stride+poolCol));
					poolOut.slice(0, ch).put(r, c, subMat.max());
					weight.w.slice(0, ch).put(r, c, subMat.argmax());
				}
			}
		}
		return poolOut;

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
					int subIdx = (int) weight.w.slice(0, ch).get(or, oc);
					propDelta.slice(0, ch).put(or*stride+subIdx%poolRow, oc*stride+subIdx/poolCol,
							error.slice(0, ch).get(or, oc));
				}
			}
		}
		return propDelta;
	}
}
