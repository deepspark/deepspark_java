package org.acl.deepspark.nn.layers;

import org.acl.deepspark.data.Tensor;
import org.acl.deepspark.data.Weight;
import org.acl.deepspark.nn.conf.LayerConf;
import org.acl.deepspark.nn.functions.Activator;
import org.acl.deepspark.nn.functions.ActivatorFactory;
import org.acl.deepspark.nn.functions.ActivatorType;
import org.acl.deepspark.utils.ArrayUtils;

import java.io.Serializable;

public class ConvolutionLayer extends BaseLayer implements Serializable {
	private int numFilter;
	private int dimRow,dimCol;
	private Activator activator;
	private static final long serialVersionUID = 140807767171115076L;

	public ConvolutionLayer(int[] shape, LayerConf conf) {
		super(shape);
		numFilter = (Integer) conf.get("numFilters");
		dimRow = (Integer) conf.get("filterRow");
		dimCol= (Integer) conf.get("filterCol");
		activator = ActivatorFactory.get((ActivatorType) conf.get("activator"));
	}

		@Override
	public Weight createWeight(LayerConf conf, int[] input) {
		// weight - 4D ( kernels, channels, x, y )
		// bias - 1D ( filter )
		// input - 3D ( channel, x, y)
		Weight w = new Weight();
		int[] dimW = new int[4];
		dimW[0] = numFilter;
		dimW[1] = input[0];
		dimW[2] = dimRow; // x
		dimW[3] = dimCol; // y

		double f_in = 1.0;
		for (int i = 0 ; i < dimW.length; i++) {
			f_in *= dimW[i];
		}

		w.w = Tensor.randn(dimW).muli(Math.sqrt(2.0/f_in));
		w.b = Tensor.zeros(dimW[0]);

		return w;
	}

	@Override
	public Tensor generateOutput(Weight weight, Tensor input) {
		int[] kernelDim = weight.getWeightShape(); // 0: # of kernels, 1: # of channel, 2: x, 3: y;
		Tensor output = Tensor.zeros(calculateOutputDimension());
		
		// TODO: check dims(image) > dims(filter)
		for (int i = 0; i < kernelDim[0]; i++) {
			for (int j = 0; j < kernelDim[1]; j++) {
				output.slice(0, i).addi(ArrayUtils.convolution(input.slice(0, j), weight.w.slice(i, j), ArrayUtils.VALID_CONV)); // valid conv
			}
			output.slice(0, i).addi(weight.b.slice(0, 0).get(i));
		}
		return output;
	}

	@Override
	public Weight gradient(Tensor input, Tensor error) {
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
	}

	@Override
	public Tensor activate(Tensor output) {
		return activator.output(output);
	}

	@Override
	public int[] calculateOutputDimension() {
		int[] dimOut = new int[] { numFilter,
								 getInputShape()[1] - dimRow + 1,  // x
								 getInputShape()[2] - dimCol + 1}; // y
		return dimOut;
	}


	@Override
	public Tensor deriveDelta(Tensor activated, Tensor error) {
		return error.mul(activator.derivative(activated));
	}

	@Override
	public Tensor calculateBackprop(Weight weight, Tensor error) {
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
