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
		for (int i = 0 ; i < dimW.length; i++) {
			f_in *= dimW[i];
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

/*
		int[] dimIn = getInputShape();
		int rowKernels = calculateOutputDimension()[1];
		int colKernels = calculateOutputDimension()[2];

		float[] reshapeArr = new float[dimow*dimCol*rowKernels*colKernels*dimIn[0]];
		int startPos = 0;
		for (int r = 0; r < rowKernels; r++) {
			for (int c = 0; c < colKernels; c++) {
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
		Tensor reshaped = Tensor.create(reshapeArr, new int[]{dimRow * dimCol * dimIn[0], rowKernels * colKernels});
		return weight.w.mmul(reshaped).addi(weight.b).transpose().reshape(calculateOutputDimension());
*/
/*
		// TODO: check dims(image) > dims(filter)
		for (int i = 0; i < kernelDim[0]; i++) {
			for (int j = 0; j < kernelDim[1]; j++) {
				output.slice(0, i).addi(ArrayUtils.convolution(input.slice(0, j), weight.w.slice(i, j), ArrayUtils.VALID_CONV)); // valid conv
			}
			output.slice(0, i).addi(weight.b.slice(0, 0).get(i));
		}
*/
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
		/*

		int[] dim = new int[4];
		int[] inputDim = getInputShape(); // 0: # of channel, 1: x, 2: y;
		
		// 0: # of filters, 1: # of channels, 2: x, 3: y;
		dim[0] = numFilter;
		dim[1] = inputDim[0];
		dim[2] = dimRow;
		dim[3] = dimCol;

		//error = error.reshape(numFilter, inputDim[1] - dimRow + 1, inputDim[2] - dimCol + 1);
		
		Weight w = new Weight();
		w.w = Tensor.zeros(dim);
		w.b = Tensor.zeros(numFilter);
		
		//bias
		for(int j = 0; j < numFilter; j++) {
			w.b.slice(0, 0).put(j, error.slice(0, j).sum());
		}
		
		//weight
		for(int i = 0; i < numFilter; i++) {
			for (int j = 0; j < inputDim[0]; j++) {
				w.w.slice(i,j).addi(ArrayUtils.convolution(input.slice(0, j), error.slice(0, i), ArrayUtils.VALID_CONV)); // valid conv
			}
		}
		return w;

		*/


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
		/*
		Tensor backProp = Tensor.zeros(getInputShape());
		int[] deltaShape = backProp.shape();

		// TODO: check dims(image) > dims(filter)
		for (int k = 0; k < deltaShape[0]; k++) {
			for (int ch = 0; ch < deltaShape[1]; ch++) {
				for (int i = 0; i < numFilter; i++) {
					backProp.slice(k, ch).addi(ArrayUtils.convolution(error.slice(k, i), ArrayUtils.flip(weight.w.slice(i, ch)),        // flip weight
							ArrayUtils.FULL_CONV));                // full convolution
				}
			}
		}
		return backProp;
		*/
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
/*
	@Override
	public INDArray generateOutputBatch(Weight weight, INDArray input) {
		int[] dim = new int[4]; // sampleSize, filter, x, y 
		int[] inputDim = getInputShape(); // 0: # of samples, 1: # of channel, 2: x, 3: y;
		int[] kernelDim = weight.getWeightShape(); // 0: # of channel, 1: # of filter, 2: x, 3: y;
		
		int numChannels = kernelDim[0];
		int numFilters = kernelDim[1];
		
		dim[0] = inputDim[0];
		dim[1] = numFilters;
		dim[2] = inputDim[2] - kernelDim[2] + 1;
		dim[3] = inputDim[3] - kernelDim[3] + 1;
		
		INDArray output = Nd4j.zeros(dim);
		
		// TODO: check dims(image) > dims(filter)
		for(int m = 0; m < inputDim[0];m++) {
			for(int i = 0; i < numFilters; i++) {
				for(int j = 0; j < numChannels; j++)
					output.slice(m).slice(i).addi(ArrayUtils.convolution(input.slice(m).slice(j), weight.w.slice(j).slice(i), ArrayUtils.VALID_CONV)); // valid conv
				output.slice(m).slice(i).addi(weight.b.getScalar(i));
			}
		}
		
		return output;
	}

	@Override
	public Weight gradientBatch(INDArray input, INDArray error) {
		// TODO Auto-generated method stub
		int[] dim = new int[4];
		int[] inputDim = getInputShape(); // 0: # of channel, 1: x, 2: y;
		
		// 0: # of channel, 1: # of filter, 2: x, 3: y;
		dim[0] = inputDim[0];
		dim[1] = numFilter;
		dim[2] = dimRow;
		dim[3] = dimCol;

		error = error.reshape(numFilter, inputDim[1] - dimRow + 1, inputDim[2] - dimCol + 1);
		
		Weight w = new Weight();
		w.w = Nd4j.zeros(dim);
		w.b = Nd4j.zeros(1, numFilter);
		
		int numSample = input.shape()[0];
		//bias
		for(int m = 0; m < numSample; m++)
			for(int j = 0; j < numFilter; j++) {
				w.b.addi(j, Nd4j.sum(error.slice(m).slice(j)).divi(numSample));
			}
		
		//weight
		for(int m = 0; m < numSample; m++)
			for(int i = 0; i < inputDim[0]; i++) {
				for (int j = 0; j < numFilter; j++) {
					w.w.slice(i).slice(j).addi(ArrayUtils.convolution(input.slice(m).slice(i), error.slice(m).slice(j), ArrayUtils.VALID_CONV).divi(numSample)); // valid conv
				}
			}
		return w;
	}

	@Override
	public INDArray calculateBackpropBatch(Weight weight, INDArray error) {
		int[] inputDim = getInputShape(); // 0: # of channel, 1: x, 2: y;
		int[] errDim = new int[inputDim.length+1]; // 0: # of samples, 1: # of channel, 2: x, 3: y;
		
		int numChannel = inputDim[0];
		int numSample = error.shape()[0];
		errDim[0] = numSample;
		System.arraycopy(inputDim, 0, errDim, 1, inputDim.length);
		
		INDArray output = Nd4j.zeros(errDim);

		// TODO: check dims(image) > dims(filter)
		for(int m =0; m < numSample; m++)
			for(int i = 0; i < numChannel; i++) {
				for(int j = 0; j < numFilter; j++) {
					output.slice(m).slice(i).addi(ArrayUtils.convolution(error.slice(m).slice(j),
							ArrayUtils.rot90(ArrayUtils.rot90(weight.w.slice(i).slice(j))),		// flip weight
							ArrayUtils.FULL_CONV)); 											// full conv
				}
			}
		return output;
	}
*/
}
