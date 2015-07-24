package org.acl.deepspark.nn.layers;

import java.io.Serializable;

import org.acl.deepspark.data.Weight;
import org.acl.deepspark.nn.conf.LayerConf;
import org.acl.deepspark.nn.functions.Activator;
import org.acl.deepspark.nn.functions.ActivatorFactory;
import org.acl.deepspark.nn.functions.ActivatorType;
import org.acl.deepspark.nn.layers.BaseLayer;
import org.acl.deepspark.nn.weights.WeightUtil;
import org.acl.deepspark.utils.ArrayUtils;
import org.acl.deepspark.utils.MathUtils;
import org.jblas.DoubleMatrix;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.factory.BaseNDArrayFactory;
import org.nd4j.linalg.factory.NDArrayFactory;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.util.NDArrayUtil;

public class ConvolutionLayer extends BaseLayer implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 140807767171115076L;
	private Activator activator;
	private int numFilter;
	private int dimRow,dimCol;

	public ConvolutionLayer(int[] shape, LayerConf conf) {
		super(shape);
		
		this.activator = ActivatorFactory.getActivator((ActivatorType) conf.get("activator"));
		this.numFilter = (Integer) conf.get("numFilter");
		this.dimRow = (Integer) conf.get("filterRow");
		this.dimCol= (Integer) conf.get("filterCol");
	}

	
	@Override
	public void initWeights() {
		if (W == null || bias == null) {
			W = new DoubleMatrix[numFilters][numChannels];
			prevDeltaW = new DoubleMatrix[numFilters][numChannels];
			
			bias = new double[numFilters];
			prevDeltaBias = new double[numFilters];
			
			for(int i = 0; i < numFilters; i++) {
				for(int j = 0; j < numChannels; j++) {
					W[i][j] = WeightUtil.randInitWeights(filterRows, filterCols, dimIn);
					prevDeltaW[i][j] = DoubleMatrix.zeros(filterRows, filterCols);
				}
				bias[i] = 0.0;
				prevDeltaBias[i] = 0;
			}
		}
	}

	// Convolution of multiple channel input images
	public DoubleMatrix[] convolution() {
		DoubleMatrix[] data = new DoubleMatrix[numFilters];
		// TODO: check dims(image) > dims(filter)
		for(int i = 0; i < numFilters; i++) {
			data[i] = DoubleMatrix.zeros(getOutputRows(), getOutputCols());
			for(int j = 0; j < numChannels; j++) {
				data[i].addi(MathUtils.convolution(input[j], W[i][j], MathUtils.VALID_CONV));
			}
			data[i].addi(bias[i]);
		}
		return data;
	}
	
	@Override
	public DoubleMatrix[] getOutput() {
		output = activate(convolution());
		return output;
	}
	
	@Override
	public void setDelta(DoubleMatrix[] propDelta) {
		delta = propDelta;
		for(int i = 0 ; i < delta.length; i++)
			delta[i].muli(output[i].mul(output[i].mul(-1.0).add(1.0)));
	}
	
	// TODO:
	@Override
	public DoubleMatrix[][] deriveGradientW() {
		DoubleMatrix[][] gradient = new DoubleMatrix[numFilters][numChannels];
		
		// update Weights
		for (int i = 0; i < numFilters; i++)
			for (int j = 0; j < numChannels; j++)
				gradient[i][j] = MathUtils.convolution(input[j], getDelta()[i], MathUtils.VALID_CONV);		
		return gradient;
	}

	@Override
	public void update(DoubleMatrix[][] gradW, double[] gradB) {
		for (int i = 0; i < numFilters; i++) {
			for (int j = 0; j < numChannels; j++) {				
				prevDeltaW[i][j].muli(momentumFactor);
				prevDeltaW[i][j].addi(W[i][j].mul(learningRate * decayLambda));
				prevDeltaW[i][j].addi(gradW[i][j].muli(learningRate));
				
				//prevDeltaBias[i] *= momentumFactor;
				prevDeltaBias[i] = (gradB[i]  + bias[i] * decayLambda)* learningRate;
				W[i][j].subi(prevDeltaW[i][j]);
				bias[i] -= prevDeltaBias[i];
			}
		}
	}
	
	@Override
	public DoubleMatrix[] deriveDelta() {
		if (delta == null || delta.length <= 0)
			return null;
		
		DoubleMatrix[] propDelta = new DoubleMatrix[numChannels];
		DoubleMatrix filter;
		
		// TODO: check dims(image) > dims(filter)
		for (int j = 0; j < numChannels; j++) {
			propDelta[j] = DoubleMatrix.zeros(dimRows, dimCols);
			for (int i = 0; i < numFilters; i++) {
				filter = new DoubleMatrix(W[i][j].toArray2());
				
				MathUtils.flip(filter);
				propDelta[j].addi(MathUtils.convolution(delta[i], filter, MathUtils.FULL_CONV));
			}
		}
		return propDelta;
	}

	@Override
	public void applyDropOut() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public int[] initWeights(int[] dim) {
		int[] outDim = new int[3];
		
		this.dimRows = dim[0];
		this.dimCols = dim[1];
		this.numChannels = dim[2];
		this.dimIn = dimRows * dimCols * numChannels;
		initWeights();
		
		outDim[0] = dim[0] - filterRows +1; //output row dimension
		outDim[1] = dim[1] - filterCols +1; //output col dimension
		outDim[2] = numFilters; //output channel dimension
		
		return outDim;
	}

	@Override
	public Weight createWeight(LayerConf conf, int[] input) {
		// weight - 4D ( channel, filter, x, y )
		// bias - 2D ( channel, filter )
		// input - 3D ( channel, x, y)
		Weight w = new Weight();
		int[] dimW = new int[4];
		dimW[0] = (Integer) conf.get("numChannel");
		dimW[1] = (Integer) conf.get("numFilter");
		dimW[2] = (Integer) conf.get("dimFilterX"); // x
		dimW[3] = (Integer) conf.get("dimFilterY"); // y
		
		w.w = Nd4j.randn(dimW);
		w.b = Nd4j.ones(dimW[0],dimW[1]).muli(0.01);
		
		return w;
	}

	@Override
	public INDArray generateOutput(Weight weight, INDArray input) {
		int[] dim = new int[3];
		int[] inputDim = getInputShape(); // 0: # of channel, 1: x, 2: y;
		int[] kernelDim = weight.getShape(); // 0: # of channel, 1: # of filter, 2: x, 3: y;
		
		int numChannels = kernelDim[0];
		int numFilters = kernelDim[1];
		
		dim[0] = numFilters;
		dim[1] = inputDim[1] - kernelDim[2] + 1;
		dim[2] = inputDim[2] - kernelDim[3] + 1;
		
		INDArray output = Nd4j.zeros(dim);
		
		// TODO: check dims(image) > dims(filter)
		for(int i = 0; i < numFilters; i++) {
			for(int j = 0; j < numChannels; j++) {
				output.slice(i).addi(Convolution.conv2d(input.slice(j), weight.w.slice(j).slice(i), Convolution.Type.VALID)); // valid conv
				output.slice(i).addi(weight.b.getScalar(j, i));
			}
		}
		return output;
	}

	@Override
	public Weight gradient(INDArray input, INDArray error) {
		int[] dim = new int[3];
		int[] inputDim = getInputShape(); // 0: # of channel, 1: x, 2: y;
		int[] kernelDim = weight.getShape(); // 0: # of channel, 1: # of filter, 2: x, 3: y;
		return null;
	}

	@Override
	public INDArray activate(INDArray output) {
		return activator.output(output);
	}

	@Override
	public int[] calculateOutputDimension(LayerConf conf, int[] input) {
		int[] dimW = new int[3];
		dimW[0] = (Integer) conf.get("numFilter");
		dimW[1] = getInputShape()[1]; // x
		dimW[2] = getInputShape()[2]; // y
		return dimW;
	}


	@Override
	public INDArray deriveDelta(INDArray error, INDArray output) {
		return error.mul(activator.derivative(output));
	}


	@Override
	public INDArray calculateBackprop(Weight weight, INDArray error) {
		int[] dim = new int[3];
		int[] inputDim = getInputShape(); // 0: # of channel, 1: x, 2: y;
		
		int numChannel = inputDim[0];
		
		dim[0] = numChannel;
		dim[1] = inputDim[1] + dimRow - 1;
		dim[2] = inputDim[2] + dimCol - 1;
		
		INDArray output = Nd4j.zeros(dim);
		
		
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
