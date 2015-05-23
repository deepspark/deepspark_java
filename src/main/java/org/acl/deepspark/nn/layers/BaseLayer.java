package org.acl.deepspark.nn.layers;

import org.acl.deepspark.nn.functions.Activator;
import org.acl.deepspark.nn.weights.WeightUtil;
import org.jblas.DoubleMatrix;

public abstract class BaseLayer {
	protected int nIn;
	protected int nOut;
	protected DoubleMatrix input;
	protected DoubleMatrix weight;
	protected double bias;
	
	public double dropOutRate;
	private int activationMethod = Activator.SIGMOID;
	
	public BaseLayer() { }
	
	public BaseLayer(int nIn, int nOut) {
		this.nIn = nIn;
		this.nOut = nOut;
		initWeights();
	}
	
	protected void initWeights() {		
		weight = WeightUtil.randInitWeights(nOut, nIn);
		bias = 0;
	}
	
	public void setWeight(DoubleMatrix weight) {
		this.weight = weight;
	}
	
	public DoubleMatrix getWeight() {
		return weight;
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
	
	public void applyDropOut() {
		
	}
	
	public abstract DoubleMatrix[] getOutput();
	public abstract DoubleMatrix[] update(DoubleMatrix[] deltas); 
}
