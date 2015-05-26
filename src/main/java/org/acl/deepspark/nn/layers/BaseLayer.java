package org.acl.deepspark.nn.layers;

import org.acl.deepspark.nn.functions.Activator;
import org.acl.deepspark.nn.weights.WeightUtil;
import org.jblas.DoubleMatrix;

public abstract class BaseLayer {
	protected int dimRows;
	protected int dimCols;
	protected int numChannels;
	protected int dimIn;
	protected int dimOut;
	
	protected double learningRate;
	protected DoubleMatrix[] input;
	
	public double dropOutRate;
	private int activationMethod = Activator.SIGMOID;
	
	public BaseLayer() { }
	
	public BaseLayer(DoubleMatrix input) {
		this.input = new DoubleMatrix[1];
		this.input[0] = input;
		
		this.dimRows = input.rows;
		this.dimCols = input.columns;
		this.numChannels = 1;
		this.dimIn = dimRows * dimCols * numChannels;
		this.learningRate = 1.0;
	}
	
	public BaseLayer(DoubleMatrix[] input) {
		this.input = input;

		this.dimRows = input[0].rows;
		this.dimCols = input[0].columns;
		this.numChannels = input.length;
		this.dimIn = dimRows * dimCols * numChannels;
		this.learningRate = 1.0;
	}
	
	public BaseLayer(DoubleMatrix input, int dimOut) {
		this(input);
		this.dimOut = dimOut;
	}
	
	public BaseLayer(DoubleMatrix[] input, int dimOut) {
		this(input);
		this.dimOut = dimOut;
	}
		
	public BaseLayer(int dimRows, int dimCols, int numChannels, int dimOut) {
		this.dimRows = dimRows;
		this.dimCols = dimCols;
		this.numChannels = numChannels;
		this.dimIn = dimRows * dimCols * numChannels;
		this.dimOut = dimOut;
		this.learningRate = 1.0;
	}
	
	// Apply activation
	public DoubleMatrix activate(DoubleMatrix matrix) {
		if(matrix == null)
			return null;
		switch(activationMethod) {
		case Activator.SIGMOID:
			return Activator.sigmoid(matrix);
			
		case Activator.TANH:
			return Activator.tanh(matrix);
			
		case Activator.RELU:
			return Activator.relu(matrix);
		}
		return Activator.sigmoid(matrix);
	}
		
	public DoubleMatrix[] activate(DoubleMatrix[] matrix) {
		int size = matrix.length;
		for (int i = 0; i < size; i++) {
			matrix[i] = activate(matrix[i]);
		}
		return matrix;
	}
	
	public abstract void initWeights(); 
	public abstract void applyDropOut();
	public abstract DoubleMatrix[] getOutput();
	public abstract DoubleMatrix[] deriveDelta(DoubleMatrix[] outputDelta);
	public abstract DoubleMatrix[] update(DoubleMatrix[] outputDelta);
};