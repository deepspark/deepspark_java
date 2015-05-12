package org.acl.deepspark.nn.layers;

import org.jblas.DoubleMatrix;

// Fully Connected HiddenLayer
public class HiddenLayer extends BaseLayer {
	private DoubleMatrix input;
	
	public HiddenLayer(DoubleMatrix input, int nOut) {
		this(input, input.length, nOut);
	}
	
	public HiddenLayer(DoubleMatrix input, int nIn, int nOut) {
		super(nIn, nOut);
		if(!input.isColumnVector()) {
			this.input = new DoubleMatrix(input.data);
		}
		this.input = input;
	}
	
	public HiddenLayer(double[] input, int nIn, int nOut) {
		super(nIn, nOut);
		this.input = new DoubleMatrix(input);
	}
	
	public DoubleMatrix feedForward() {
		return activate(weight.mmul(input).add(bias));
	}
	
	
	@Override
	public double[] getOutput() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void update(DoubleMatrix[] weights) {
		// TODO Auto-generated method stub
		
	}
	
}
