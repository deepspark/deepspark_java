package org.acl.deepspark.nn.layers;

import java.io.Serializable;

import org.acl.deepspark.nn.functions.Activator;
import org.jblas.DoubleMatrix;

public abstract class BaseLayer implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 6281205493824547766L;
	protected int dimRows;
	protected int dimCols;
	protected int numChannels;
	protected int dimIn;
	protected int dimOut;
	
	protected double learningRate = 0.1;
	protected DoubleMatrix[] input;
	protected DoubleMatrix[] output;
	
	protected DoubleMatrix[] delta;
	
	public double dropOutRate;
	private int activationMethod = Activator.SIGMOID;
	
	public BaseLayer(int activator) {
		this.activationMethod = activator;
	}
	
	public BaseLayer() { }
	
	public BaseLayer(DoubleMatrix input) {
		this.input = new DoubleMatrix[1];
		this.input[0] = input;
		
		this.dimRows = input.rows;
		this.dimCols = input.columns;
		this.numChannels = 1;
		this.dimIn = dimRows * dimCols * numChannels;
	}
	
	public void setInput(DoubleMatrix input) {
		this.input = new DoubleMatrix[1];
		this.input[0] = input;
		
		this.dimRows = input.rows;
		this.dimCols = input.columns;
		this.numChannels = 1;
		this.dimIn = dimRows * dimCols * numChannels;
		
		initWeights();
	}
	
	public BaseLayer(DoubleMatrix[] input) {
		this.input = input;

		this.dimRows = input[0].rows;
		this.dimCols = input[0].columns;
		this.numChannels = input.length;
		this.dimIn = dimRows * dimCols * numChannels;
	}
	
	public void setInput(DoubleMatrix[] input) {
		this.input = input;
		
		this.dimRows = input[0].rows;
		this.dimCols = input[0].columns;
		this.numChannels = input.length;
		this.dimIn = dimRows * dimCols * numChannels;
		
		initWeights();
	}
	
	public BaseLayer(DoubleMatrix input, int dimOut) {
		this(input);
		this.dimOut = dimOut;
	}
	
	public BaseLayer(DoubleMatrix[] input, int dimOut) {
		this(input);
		this.dimOut = dimOut;
	}

	// Apply activation
	public DoubleMatrix activate(DoubleMatrix matrix) {
		DoubleMatrix activated = null;
		switch(activationMethod) {
		case Activator.SIGMOID:
			activated = Activator.sigmoid(matrix);
			break;
		case Activator.TANH:
			activated = Activator.tanh(matrix);
			break;
		case Activator.RELU:
			activated = Activator.relu(matrix);
			break;
		case Activator.SOFTMAX:
			activated = Activator.softmax(matrix);
		}
		return activated;
	}
		
	public DoubleMatrix[] activate(DoubleMatrix[] matrices) {
		for (DoubleMatrix matrix : matrices)
			matrix = activate(matrix);
		return matrices;
	}
	
	public DoubleMatrix[] getDelta() {
		return delta;
	}
	public abstract void setDelta(DoubleMatrix[] delta);
	
	public abstract void initWeights(); 
	public abstract int[] initWeights(int[] dim);
	public abstract void applyDropOut();
	public abstract int[] getWeightInfo();
	public abstract DoubleMatrix[][] deriveGradientW();
	public abstract DoubleMatrix[] getOutput();
	public abstract DoubleMatrix[] deriveDelta();
	public abstract void update(DoubleMatrix[][] gradW, double[] gradB);
};