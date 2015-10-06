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
import org.jblas.FloatMatrix;
import org.jblas.ranges.RangeUtils;

import java.io.Serializable;

public class ConvolutionLayer extends BaseLayer implements Serializable {
	private int kernels;
	private int kernelRow,kernelCol;
	private int stride;
	private int padding;
	private boolean gpuAccel;
	private Activator activator;

	private static final long serialVersionUID = 140807767171115076L;

	public ConvolutionLayer(int[] shape, LayerConf conf, boolean gpuAccel) {
		super(shape);
		kernels = (Integer) conf.get("num_output");
		kernelRow = (Integer) conf.get("kernel_row");
		kernelCol= (Integer) conf.get("kernel_col");
		stride = (Integer) conf.get("stride");
		padding = (Integer) conf.get("zeroPad");
		activator = ActivatorFactory.get((ActivatorType) conf.get("activator"));
		this.gpuAccel = gpuAccel;
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

	return new Weight  (WeightFactory.create(typeW, valueW, kernelRow*kernelCol*input[1], kernels),
						WeightFactory.create(typeB, valueB, kernels));
	}

	@Override
	public Tensor generateOutput(Weight weight, Tensor input) {
		int channels = getDimIn()[1];
		int rowKernels = calcOutputShape()[2];
		int colKernels = calcOutputShape()[3];

		float[] reshapeArr = new float[kernelRow*kernelCol*channels*rowKernels*colKernels];
		int startPos = 0;

		/* reshaping to matrix to simplify convolution to normal matrix multiplication */
		input = ArrayUtils.zeroPad(input, padding);
		for (int c = 0; c < colKernels; c++) {
			for (int r = 0; r < rowKernels; r++) {
				for (int ch = 0; ch < channels; ch++) {
					System.arraycopy(input.slice(0, ch).get(RangeUtils.interval(r*stride, r*stride+kernelRow), RangeUtils.interval(c*stride, c*stride+kernelCol)).toArray(),
							0,
							reshapeArr,
							startPos,
							kernelRow*kernelCol);
					startPos += kernelRow*kernelCol;
				}
			}
		}

		Tensor reshaped = Tensor.create(reshapeArr, new int[]{kernelRow*kernelCol*channels, rowKernels*colKernels}).transpose();
		return reshaped.mmul(weight.w, gpuAccel).addiRowTensor(weight.b).reshape(kernels, rowKernels, colKernels);
	}

	@Override
	public Weight gradient(Tensor input, Tensor error) {
		int channels = getDimIn()[1];
		int rowKernels = calcOutputShape()[2];
		int colKernels = calcOutputShape()[3];

		float[] reshapeArr = new float[kernelRow*kernelCol*channels*rowKernels*colKernels];
		int startPos = 0;

		/* reshaping to matrix to simplify convolution to normal matrix multiplication */
		input = ArrayUtils.zeroPad(input, padding);
		for (int c = 0; c < colKernels; c++) {
			for (int r = 0; r < rowKernels; r++) {
				for (int ch = 0; ch < channels; ch++) {
					System.arraycopy(input.slice(0, ch).get(RangeUtils.interval(r*stride, r*stride+kernelRow), RangeUtils.interval(c*stride, c*stride+kernelCol)).toArray(),
							0,
							reshapeArr,
							startPos,
							kernelRow*kernelCol);
					startPos += kernelRow*kernelCol;
				}
			}
		}

		Tensor reshaped = Tensor.create(reshapeArr, new int[]{kernelRow*kernelCol*channels, rowKernels*colKernels});
		error = error.reshape(rowKernels*colKernels, kernels);

		return new Weight(reshaped.mmul(error, gpuAccel), error.columnSums());
	}

	@Override
	public Tensor activate(Tensor output) {
		return activator.output(output);
	}

	@Override
	public int[] calcOutputShape() {
		int[] dimOut = new int[] {
				getDimIn()[0],										// # of batch processed (currently 1)
				kernels,											// # of featureMaps
				(getDimIn()[2]-kernelRow+2*padding)/stride + 1,  	// featureMap width
				(getDimIn()[3]-kernelCol+2*padding)/stride + 1}; 	// featureMap height
		return dimOut;
	}


	@Override
	public Tensor deriveDelta(Tensor activated, Tensor error) {
		return error.mul(activator.derivative(activated));
	}

	@Override
	public Tensor calculateBackprop(Weight weight, Tensor error) {

		int[] dimIn = getDimIn();
		int rowKernels = calcOutputShape()[2];
		int colKernels = calcOutputShape()[3];
		error = error.reshape(rowKernels*colKernels, kernels).transpose();

		float[] backPropArr = weight.w.mmul(error, gpuAccel).toArray();
		float[] reshapeArr = new float[kernelRow*kernelCol];
		int startPos = 0;

		Tensor backProp = Tensor.zeros(dimIn[0], dimIn[1], dimIn[2]+2*padding, dimIn[3]+2*padding);
		for (int c = 0; c < colKernels; c++) {
			for (int r = 0; r < rowKernels; r++) {
				for (int ch = 0; ch < dimIn[1]; ch++) {
					System.arraycopy(backPropArr, startPos, reshapeArr, 0, kernelRow * kernelCol);
					backProp.slice(0, ch).put(RangeUtils.interval(r*stride, r*stride+kernelRow), RangeUtils.interval(c*stride, c*stride+kernelCol), new FloatMatrix(kernelRow, kernelCol, reshapeArr));
					startPos += kernelRow*kernelCol;
				}
			}
		}
		return ArrayUtils.centerCrop(backProp, padding);
	}
}
