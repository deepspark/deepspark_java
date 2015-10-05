package org.acl.deepspark.nn.layers;

import org.acl.deepspark.data.Tensor;
import org.acl.deepspark.data.Weight;
import org.acl.deepspark.nn.conf.LayerConf;
import org.jblas.FloatMatrix;
import org.jblas.ranges.RangeUtils;

import java.io.Serializable;

public class PoolingLayer extends BaseLayer implements Serializable, Layer {
	/**
	 * 
	 */

	private int kernelRow;
	private int kernelCol;
	private int stride;

	private static final long serialVersionUID = -4318643106939173007L;

	public PoolingLayer(int[] inputShape, LayerConf conf) {
		super(inputShape);
		this.kernelRow = (Integer) conf.get("kernel_row");
		this.kernelCol = (Integer) conf.get("kernel_col");
		this.stride = (Integer) conf.get("stride");
	}

	@Override
	public Weight createWeight(LayerConf conf, int[] input) {
		Weight weight = new Weight();
		weight.w = Tensor.zeros(calcOutputShape());
		weight.b = Tensor.zeros(1);
		return weight;
	}

	@Override
	public int[] calcOutputShape() {
		int[] dimIn = getDimIn();
		return new int[] {dimIn[0], dimIn[1], (dimIn[2]- kernelRow)/stride+1, (dimIn[3]- kernelCol)/stride+1};
	}

	@Override
	public Tensor generateOutput(Weight weight, Tensor input) {
		int channels = getDimIn()[1];
		int rowKernels = calcOutputShape()[2];
		int colKernels = calcOutputShape()[3];

		FloatMatrix subMat;
		Tensor poolOut = Tensor.zeros(calcOutputShape());

		for (int ch = 0; ch < channels; ch++) {
			for (int c = 0; c < colKernels; c++) {
				for (int r = 0; r < rowKernels; r++) {
					subMat = input.slice(0, ch).get(RangeUtils.interval(r*stride, r*stride+ kernelRow), RangeUtils.interval(c*stride, c*stride+ kernelCol));
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
		Tensor propDelta = Tensor.zeros(getDimIn());
		int[] dimOut = calcOutputShape();
		for (int ch = 0; ch < dimOut[1]; ch++) {
			for (int or = 0; or < dimOut[2]; or++) {
				for (int oc = 0; oc < dimOut[3]; oc++) {
					int subIdx = (int) weight.w.slice(0, ch).get(or, oc);
					propDelta.slice(0, ch).put(or*stride+subIdx% kernelRow, oc*stride+subIdx/ kernelCol,
							propDelta.slice(0, ch).get(or*stride+subIdx% kernelRow, oc*stride+subIdx/ kernelCol) + error.slice(0, ch).get(or, oc));
				}
			}
		}
		return propDelta;
	}
}
