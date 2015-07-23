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

	private Activator activator;

	public PoolingLayer(int[] inputShape, ActivatorType t) {
		super(inputShape);
		activator = ActivatorFactory.getActivator(t);
	}

	// complete //
	@Override
	 public Weight createWeight(LayerConf conf, int[] input) {
		return null;
	}

	// complete // TODO: dimensionality check
	@Override
	public int[] calculateOutputDimension(LayerConf conf, int[] input) {
		poolRow = (int) conf.get("poolRow");
		poolCol = (int) conf.get("poolCol");
		return new int[] {input[0], input[1], poolRow, poolCol};
	}

	// complete // TODO: Test needed
	@Override
	public INDArray generateOutput(Weight weight, INDArray input) {
		int numFilter = input.size(0);
		int numChannel = input.size(1);
		int rows = input.size(2);
		int cols = input.size(3);

		double value;
		double outValue;

		INDArray output = Nd4j.create(numFilter, numChannel, (rows/poolRow), (cols/poolCol));
		maskArray = Nd4j.create(numFilter, numChannel, (rows/poolRow), (cols/poolCol));
		for (int i = 0 ; i < numFilter; i++) {
			for (int j = 0; j < numChannel; j++) {
				for (int r = 0; r < rows; r++) {
					int or = r / poolRow;
					for (int c = 0; c < cols; c++) {
						int oc = c / poolCol;
						value = input.getDouble(i, j, r, c);
						outValue = output.getDouble(i, j, or, oc);
						if (value > outValue) {
							output.putScalar(new int[]{i, j, or, oc}, value);
						//	maskArray.putScalar(new int[]{i, j, or, oc}, input.index(r, c));
						}
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

	// complete // TODO: dimensionality check
	@Override
	public INDArray deriveDelta(INDArray error, INDArray output) {
		return error.mul(activator.derivative(output));
	}

	// complete //
	@Override
	public Weight gradient(INDArray input, INDArray error) {
		return null;
	}


	// TODO: Test Needed
	@Override
	public INDArray calculateBackprop(Weight weight, INDArray error) {
		INDArray delta = Nd4j.create(getInputShape());
		int numFilter = delta.size(0);
		int numChannel = delta.size(1);
		int rows = maskArray.size(2);
		int cols = maskArray.size(3);

		for (int i = 0 ;i < numFilter; i++) {
			for (int j = 0; j < numChannel; j++) {
				for (int or = 0; or < rows; or++) {
					for (int oc = 0; oc < cols; oc++) {
						INDArray array = delta.slice(i).slice(j);
						array.putScalar(maskArray.getInt(i, j, or, oc), error.getDouble(i, j, or, oc));
					}
				}
			}
		}
		return delta;
	}
}
