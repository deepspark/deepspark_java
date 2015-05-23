package org.acl.deepspark.nn.layers;

import org.jblas.DoubleMatrix;

// Fully Connected HiddenLayer
public class FullyConnLayer extends BaseLayer {
	private DoubleMatrix input;
	
	public FullyConnLayer(DoubleMatrix input, int nOut) {
		this(input, input.length, nOut);
	}
	
	public FullyConnLayer(DoubleMatrix input, int nIn, int nOut) {
		super(nIn, nOut);
		if(!input.isColumnVector()) {
			this.input = new DoubleMatrix(input.data);
		}
		this.input = input;
	}
	
	public FullyConnLayer(double[] input, int nIn, int nOut) {
		super(nIn, nOut);
		this.input = new DoubleMatrix(input);
	}
	
	public void feedForward() {
		
	}
	
	
	@Override
	public DoubleMatrix[] getOutput() {
		if (input != null && weight != null) {
			DoubleMatrix[] matrix = new DoubleMatrix[1];
			matrix[0] = weight.mmul(input);
			return matrix;
		}
		return null;
	}

	@Override
	public DoubleMatrix[] update(DoubleMatrix[] deltas) {
		return deltas;
		// TODO Auto-generated method stub
		
	}

	
}
