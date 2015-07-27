package org.acl.deepspark.nn.layers;

import java.io.Serializable;

import org.acl.deepspark.data.Weight;
import org.acl.deepspark.nn.conf.LayerConf;
import org.acl.deepspark.nn.functions.Activator;
import org.acl.deepspark.nn.functions.ActivatorFactory;
import org.acl.deepspark.nn.functions.ActivatorType;
import org.acl.deepspark.nn.layers.BaseLayer;
import org.acl.deepspark.utils.ArrayUtils;
import org.jblas.DoubleMatrix;
import org.jblas.ranges.RangeUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

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

	// complete //
	@Override
	 public Weight createWeight(LayerConf conf, int[] input) {
		return null;
	}

	// complete //
	@Override
	public int[] calculateOutputDimension(LayerConf conf, int[] input) {
		return new int[] {input[0], input[1]/poolRow, input[2]/poolCol};
	}

	// complete //
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

	// complete //
	@Override
	public INDArray activate(INDArray output) {
		return activator.output(output);
	}

	// complete //
	@Override
	public INDArray deriveDelta(INDArray output, INDArray error) {
		return error.mul(activator.derivative(output));
	}

	// complete //
	@Override
	public Weight gradient(INDArray input, INDArray error) {
		return null;
	}


	// complete
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
}
