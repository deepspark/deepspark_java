package org.acl.deepspark.nn.layers;

import org.acl.deepspark.nn.params.Activations;
import org.acl.deepspark.nn.weights.WeightInit;
import org.jblas.DoubleMatrix;

public abstract class BaseLayer {
	protected int nIn;
	protected int nOut;
	protected DoubleMatrix weight;
	protected double bias;
	
	public double dropOutRate;
	private int activationMethod = Activations.SIGMOID;
	
	public BaseLayer() { }
	
	public BaseLayer(int nIn, int nOut) {
		this.nIn = nIn;
		this.nOut = nOut;
		this.weight = initRandWeights();
	}
	
	
	
	protected DoubleMatrix initRandWeights() {
		int fanIn;
		int fanOut;
		
		return DoubleMatrix.rand(nOut, nIn);
/*		switch(activationMethod) {
		
		// TODO: modify weights elements
		case Activations.SIGMOID:
			break;
			
		case Activations.TANH:
			break;
			
		case Activations.RELU:
			break;
		}*/
	}
	
	public DoubleMatrix getWeight() {
		return weight;
	}
	
	// Apply Sigmoid activation
	// TODO : add tanh, RELU activation
	public DoubleMatrix activate(DoubleMatrix matrix) {
		double activation;
		for(int m = 0; m < matrix.rows; m++) {
			for(int n = 0; n < matrix.columns; n++) {
				activation = 1.0 / (1.0 + Math.exp(-1.0 * matrix.get(m, n)));
				matrix.put(m, n, activation);
			}
		}
		return matrix;
	}
		
	public DoubleMatrix[] activate(DoubleMatrix[] matrix) {
		int size = matrix.length;
		for (int i = 0; i < size; i++) {
			matrix[i] = activate(matrix[i]);
		}
		return matrix;
	}
	
	public abstract double[] getOutput();
	public abstract void update(DoubleMatrix[] weights); 
}
