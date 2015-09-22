package org.acl.deepspark.nn.layers;

import org.acl.deepspark.data.Tensor;
import org.acl.deepspark.data.Weight;
import org.acl.deepspark.nn.conf.LayerConf;
import org.acl.deepspark.nn.functions.Activator;
import org.acl.deepspark.nn.functions.ActivatorFactory;
import org.acl.deepspark.nn.functions.ActivatorType;
import org.acl.deepspark.utils.ArrayUtils;
import org.jblas.DoubleMatrix;
import org.jblas.FloatMatrix;
import org.jblas.ranges.RangeUtils;

import java.io.Serializable;

public class ConvolutionLayer extends BaseLayer implements Serializable {
	private int numFilter;
	private int dimRow,dimCol;
	private int stride;
	private int padding;
	private Activator activator;
	private static final long serialVersionUID = 140807767171115076L;

	public ConvolutionLayer(int[] shape, LayerConf conf) {
		super(shape);
		numFilter = (Integer) conf.get("numFilters");
		dimRow = (Integer) conf.get("filterRow");
		dimCol= (Integer) conf.get("filterCol");
		stride = (Integer) conf.get("stride");
		padding = (Integer) conf.get("padding");
		activator = ActivatorFactory.get((ActivatorType) conf.get("activator"));
	}

		@Override
	public Weight createWeight(LayerConf conf, int[] input) {
		// weight - 4D ( kernels, channels, x, y )
		// bias - 1D ( filter )
		// input - 3D ( channel, x, y)
		int[] dimW = new int[] {dimRow*dimCol*input[0], numFilter};

		double f_in = 1.0;
		for (int i = 0 ; i < input.length; i++) {
			f_in *= input[i];
		}

		Weight w = new Weight();
		w.w = Tensor.randn(dimW).muli((float) Math.sqrt(2.0/f_in));
		w.b = Tensor.zeros(numFilter);

		return w;
	}

	@Override
	public Tensor generateOutput(Weight weight, Tensor input) {
		int[] dimIn = getInputShape();
		int rowKernels = calculateOutputDimension()[1];
		int colKernels = calculateOutputDimension()[2];

		float[] reshapeArr = new float[dimRow*dimCol*dimIn[0]*rowKernels*colKernels];
		int startPos = 0;

		/* reshaping to matrix to simplify convolution to normal matrix multiplication */
		for (int c = 0; c < colKernels; c++) {
			for (int r = 0; r < rowKernels; r++) {
				for (int ch = 0; ch < dimIn[0]; ch++) {
					System.arraycopy(input.slice(0, ch).get(RangeUtils.interval(r*stride, r*stride+dimRow), RangeUtils.interval(c*stride, c*stride+dimCol)).toArray(),
							0,
							reshapeArr,
							startPos,
							dimRow*dimCol);
					startPos += dimRow*dimCol;
				}
			}
		}

		Tensor reshaped = Tensor.create(reshapeArr, new int[]{dimRow*dimCol*dimIn[0], rowKernels*colKernels}).transpose();
		return reshaped.mmul(weight.w).addiRowTensor(weight.b).reshape(numFilter, rowKernels, colKernels);
	}

	@Override
	public Weight gradient(Tensor input, Tensor error) {
		int[] dimIn = getInputShape();
		int rowKernels = calculateOutputDimension()[1];
		int colKernels = calculateOutputDimension()[2];

		float[] reshapeArr = new float[dimRow*dimCol*dimIn[0]*rowKernels*colKernels];
		int startPos = 0;

		/* reshaping to matrix to simplify convolution to normal matrix multiplication */
		for (int c = 0; c < colKernels; c++) {
			for (int r = 0; r < rowKernels; r++) {
				for (int ch = 0; ch < dimIn[0]; ch++) {
					System.arraycopy(input.slice(0, ch).get(RangeUtils.interval(r*stride, r*stride+dimRow), RangeUtils.interval(c*stride, c*stride+dimCol)).toArray(),
							0,
							reshapeArr,
							startPos,
							dimRow*dimCol);
					startPos += dimRow*dimCol;
				}
			}
		}

		Tensor reshaped = Tensor.create(reshapeArr, new int[]{dimRow*dimCol*dimIn[0], rowKernels*colKernels});
		error = error.reshape(rowKernels*colKernels, numFilter);

		return new Weight(reshaped.mmul(error), error.columnSums());
	}

	@Override
	public Tensor activate(Tensor output) {
		return activator.output(output);
	}

	@Override
	public int[] calculateOutputDimension() {
		int[] dimOut = new int[] { numFilter,						// # of featureMaps
				(getInputShape()[1]-dimRow+2*padding)/stride + 1,  	// featureMap width
				(getInputShape()[2]-dimCol+2*padding)/stride + 1}; 	// featureMap height
		return dimOut;
	}


	@Override
	public Tensor deriveDelta(Tensor activated, Tensor error) {
		return error.mul(activator.derivative(activated));
	}

	@Override
	public Tensor calculateBackprop(Weight weight, Tensor error) {

		int[] dimIn = getInputShape();
		int rowKernels = calculateOutputDimension()[1];
		int colKernels = calculateOutputDimension()[2];
		error = error.reshape(rowKernels*colKernels, numFilter).transpose();

		float[] backPropArr = weight.w.mmul(error).toArray();
		float[] reshapeArr = new float[dimRow*dimCol];
		int startPos = 0;

		Tensor backProp = Tensor.zeros(getInputShape());
		for (int c = 0; c < colKernels; c++) {
			for (int r = 0; r < rowKernels; r++) {
				for (int ch = 0; ch < dimIn[0]; ch++) {
					System.arraycopy(backPropArr, startPos, reshapeArr, 0, dimRow*dimCol);
					backProp.slice(0, ch).put(RangeUtils.interval(r*stride, r*stride+dimRow), RangeUtils.interval(c*stride, c*stride+dimCol), new FloatMatrix(dimRow, dimCol, reshapeArr));
					startPos += dimRow*dimCol;
				}
			}
		}
		return backProp;
	}
}
