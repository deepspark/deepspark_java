package org.acl.deepspark.nn.layers.cnn;

import java.io.Serializable;

import org.acl.deepspark.nn.layers.BaseLayer;
import org.acl.deepspark.nn.weights.WeightUtil;
import org.acl.deepspark.utils.MathUtils;
import org.jblas.DoubleMatrix;

public class ConvolutionLayer extends BaseLayer  implements Serializable{
	/**
	 * 
	 */
	private static final long serialVersionUID = 140807767171115076L;
	private int filterRows, filterCols, numFilters; // filter spec.
	private DoubleMatrix[][] W; // filterId, x, y
	private double[] bias;
	
	// momentum 
	private double momentumFactor = 0.0;
	private DoubleMatrix[][] prevDeltaW;
	private double[] prevDeltaBias;
		
	// weight decay
	private double decayLambda = 1e-5;
	
	private int[] stride = {1, 1};
	private int zeroPadding = 0;
	private boolean useZeroPadding = true;

	public ConvolutionLayer(int filterRows, int filterCols, int numFilters) {
		super();
		this.filterRows = filterRows;
		this.filterCols = filterCols;
		this.numFilters = numFilters;
	}
	
	public ConvolutionLayer(DoubleMatrix input, int filterRows, int filterCols, int numFilters) {
		super(input);
		this.filterRows = filterRows;
		this.filterCols = filterCols;
		this.numFilters = numFilters;
		initWeights();
	}
	
	public ConvolutionLayer(DoubleMatrix[] input, int filterRows, int filterCols, int numFilters) {
		super(input);
		this.filterRows = filterRows;
		this.filterCols = filterCols;
		this.numFilters = numFilters;
		initWeights();
	}
	
	public ConvolutionLayer(DoubleMatrix input, int filterRows, int filterCols, int numFilters, double momentum) {
		this(input, filterRows, filterCols, numFilters);
		momentumFactor = momentum;
	}
	
	public ConvolutionLayer(DoubleMatrix[] input, int filterRows, int filterCols, int numFilters,double momentum) {
		this(input, filterRows, filterCols, numFilters);
		momentumFactor = momentum;
	}
	
	public void setFilterWeights(DoubleMatrix[][] filters) {
		W = filters;
	}
	
	public DoubleMatrix[][] getFilterWeights() {
		return W;
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
					W[i][j] = WeightUtil.randInitWeights(filterRows, filterCols);
					prevDeltaW[i][j] = DoubleMatrix.zeros(filterRows, filterCols);
				}
				bias[i] = 0.01;
				prevDeltaBias[i] = 0;
			}
		}
	}

	public int getNumOfChannels() {
		return numChannels;
	}
	
	public int getNumOfFilter() {
		return numFilters;
	}
	
	private int getOutputRows() {
		return  dimRows - filterRows + 1;
	}
	
	private int getOutputCols() {
		return  dimCols - filterCols + 1;
	}
	
	// Convolution of multiple channel input images
	public DoubleMatrix[] convolution() {
		DoubleMatrix[] data = new DoubleMatrix[numFilters];
		// TODO: check dims(image) > dims(filter)
		for(int i = 0; i < numFilters; i++) {
			for(int j = 0; j < numChannels; j++) {
				data[i] = MathUtils.convolution(input[j], W[i][j], MathUtils.VALID_CONV);
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
			delta[i].mul(output[i].mul(output[i].mul(-1.0).add(1.0)));
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
				
				prevDeltaBias[i] *= momentumFactor;
				prevDeltaBias[i] += (gradB[i]  + bias[i] * decayLambda)* learningRate;
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
			propDelta[j] = new DoubleMatrix(dimRows, dimCols);
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
}
