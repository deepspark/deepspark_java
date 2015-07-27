package org.acl.deepspark.nn.layers;

import java.io.Serializable;

import org.acl.deepspark.data.Weight;
import org.acl.deepspark.nn.conf.LayerConf;
import org.acl.deepspark.utils.ArrayUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.factory.Nd4j;

public class ConvolutionLayer extends BaseLayer implements Serializable {
	private static final long serialVersionUID = 140807767171115076L;
	private int numFilter;
	private int dimRow,dimCol;

	public ConvolutionLayer(int[] shape, LayerConf conf) {
		super(shape, conf);
		this.numFilter = (Integer) conf.get("numFilter");
		this.dimRow = (Integer) conf.get("filterRow");
		this.dimCol= (Integer) conf.get("filterCol");
	}

		@Override
	public Weight createWeight(LayerConf conf, int[] input) {
		// weight - 4D ( channel, filter, x, y )
		// bias - 2D ( channel, filter )
		// input - 3D ( channel, x, y)
		Weight w = new Weight();
		int[] dimW = new int[4];
		dimW[0] = input[0];
		dimW[1] = (Integer) conf.get("numFilter");
		dimW[2] = (Integer) conf.get("filterRow"); // x
		dimW[3] = (Integer) conf.get("filterCol"); // y
		
		w.w = Nd4j.randn(dimW);
		w.b = Nd4j.ones(1, dimW[1]).muli(0.01);
		
		return w;
	}

	@Override
	public INDArray generateOutput(Weight weight, INDArray input) {
		int[] dim = new int[3];
		int[] inputDim = getInputShape(); // 0: # of channel, 1: x, 2: y;
		int[] kernelDim = weight.getWeightShape(); // 0: # of channel, 1: # of filter, 2: x, 3: y;
		
		int numChannels = kernelDim[0];
		int numFilters = kernelDim[1];
		
		dim[0] = numFilters;
		dim[1] = inputDim[1] - kernelDim[2] + 1;
		dim[2] = inputDim[2] - kernelDim[3] + 1;
		
		INDArray output = Nd4j.zeros(dim);
		
		// TODO: check dims(image) > dims(filter)
		for(int i = 0; i < numFilters; i++) {
			for(int j = 0; j < numChannels; j++)
				output.slice(i).addi(Convolution.conv2d(input.slice(j), weight.w.slice(j).slice(i), Convolution.Type.VALID)); // valid conv
			output.slice(i).addi(weight.b.getScalar(i));
		}
		return output;
	}

	@Override
	public Weight gradient(INDArray input, INDArray error) {
		int[] dim = new int[4];
		int[] inputDim = getInputShape(); // 0: # of channel, 1: x, 2: y;
		
		// 0: # of channel, 1: # of filter, 2: x, 3: y;
		dim[0] = inputDim[0];
		dim[1] = numFilter; 
		dim[2] = dimRow;
		dim[3] = dimCol;
		
		error.reshape(numFilter, inputDim[1] - dimRow + 1, inputDim[2] - dimCol + 1);
		
		Weight w = new Weight();
		w.w = Nd4j.zeros(dim);
		w.b = Nd4j.zeros(1, numFilter);
		
		//bias
		for(int j = 0; j < numFilter; j++) {
			w.b.put(j, Nd4j.sum(error.slice(j)));
		}
		
		//weight
		for(int i = 0; i < inputDim[0]; i++) {
			for(int j = 0; j < numFilter; j++) {
				w.w.slice(i).slice(j).addi(Convolution.conv2d(input.slice(i), error.slice(j), Convolution.Type.VALID)); // valid conv
			}
		}
		
		return w;
	}

	@Override
	public INDArray activate(INDArray output) {
		return activator.output(output);
	}

	@Override
	public int[] calculateOutputDimension(LayerConf conf, int[] input) {
		int[] dimW = new int[3];
		dimW[0] = (Integer) conf.get("numFilter");
		dimW[1] = getInputShape()[1] - dimRow + 1; // x
		dimW[2] = getInputShape()[2] - dimCol + 1; // y
		return dimW;
	}


	@Override
	public INDArray deriveDelta(INDArray output, INDArray error) {
		return error.mul(activator.derivative(output));
	}


	@Override
	public INDArray calculateBackprop(Weight weight, INDArray error) {
		int[] inputDim = getInputShape(); // 0: # of channel, 1: x, 2: y;
		int numChannel = inputDim[0];
		
		INDArray output = Nd4j.zeros(inputDim);

		// TODO: check dims(image) > dims(filter)
		for(int i = 0; i < numChannel; i++) {
			for(int j = 0; j < numFilter; j++) {
				output.slice(i).addi(Convolution.conv2d(error.slice(j), 
						ArrayUtils.rot90(ArrayUtils.rot90(weight.w.slice(i).slice(j))),		// flip weight
						Convolution.Type.FULL)); 											// full conv
			}
		}
		return output;
	}
}
