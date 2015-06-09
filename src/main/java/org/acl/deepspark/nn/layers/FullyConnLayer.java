package org.acl.deepspark.nn.layers;

import org.acl.deepspark.nn.weights.WeightUtil;
import org.jblas.DoubleMatrix;

// Fully Connected HiddenLayer
public class FullyConnLayer extends BaseLayer {
	private DoubleMatrix W;
	private DoubleMatrix output;
	private double momentumFactor = 0.95;
	private DoubleMatrix prevDeltaW;
	private double prevDeltaBias;
	private double bias = 0.01;
	private double decayLambda = 1e-5;

	public FullyConnLayer(int nOut) {
		this.dimOut = nOut;
	}
	
	public FullyConnLayer(DoubleMatrix input, int nOut) {
		super(input, nOut);
		initWeights();
	}
	
	public FullyConnLayer(DoubleMatrix[] input, int nOut) {
		super(input, nOut);
		initWeights();
	}
	
	
	public FullyConnLayer(DoubleMatrix input, int nOut, double momentum) {
		this(input, nOut);
		momentumFactor = momentum;
	}
	
	public FullyConnLayer(DoubleMatrix[] input, int nOut, double momentum) {
		this(input, nOut);
		momentumFactor = momentum;
	}
	
	public void setWeight(DoubleMatrix weight) {
		this.W = weight;
	}
	
	public DoubleMatrix getWeight() {
		return W;
	}
	
	@Override
	public DoubleMatrix[] getOutput() {
		// output: dimOut x 1 column Vector
		output = activate(W.mmul(WeightUtil.flat2Vec(input)).add(bias));
		DoubleMatrix[] postActivation = { output };
		return postActivation;
	}
	
	@Override
	public DoubleMatrix[] deriveDelta(DoubleMatrix[] delta) {
		double[] input = delta[0].transpose().mmul(W).toArray();
		DoubleMatrix[] propDelta = new DoubleMatrix[numChannels];
		
		int index = 0;
		for (int i = 0; i < numChannels; i++) {
			propDelta[i] = new DoubleMatrix(dimRows, dimCols);
			for (int n = 0; n < dimCols; n++) {
				for (int m = 0; m < dimRows; m++) {
					propDelta[i].put(m, n, input[index++]);
				}
			}
		}
		return propDelta;
	}

	@Override
	public DoubleMatrix[] update(DoubleMatrix[] propDelta) {
		propDelta[0].muli(output.mul(output.mul(-1.0).add(1.0)));
		
		prevDeltaW.muli(momentumFactor);
		prevDeltaW.addi(propDelta[0].mmul(WeightUtil.flat2Vec(input).transpose()).mul(learningRate));
		prevDeltaW.addi(W.mul(learningRate * decayLambda ));
		prevDeltaBias = propDelta[0].sum() * learningRate + prevDeltaBias * momentumFactor; 
		 
		// weight update
		W.subi(prevDeltaW);
		bias -= prevDeltaBias;
		// propagate delta to the previous layer
		return deriveDelta(propDelta);
	}

	@Override
	public void initWeights() {
		if (W == null) {
			W = WeightUtil.randInitWeights(dimOut, dimIn);
			prevDeltaW = DoubleMatrix.zeros(dimOut, dimIn);
			prevDeltaBias = 0;
		}
	}

	@Override
	public void applyDropOut() {
		// TODO Auto-generated method stub
	}

	

	
}
