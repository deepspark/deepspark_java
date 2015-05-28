package org.acl.deepspark.nn.layers;

import org.acl.deepspark.nn.weights.WeightUtil;
import org.jblas.DoubleMatrix;

// Fully Connected HiddenLayer
public class FullyConnLayer extends BaseLayer {
	private DoubleMatrix W;
	private double bias;

	public FullyConnLayer(DoubleMatrix input, int nOut) {
		super(input, nOut);
		initWeights();
	}
	
	public FullyConnLayer(DoubleMatrix[] input, int nOut) {
		super(input, nOut);
		initWeights();
	}
	
	public void setWeight(DoubleMatrix weight) {
		this.W = weight;
	}
	
	public DoubleMatrix getWeight() {
		return W;
	}
	
	// output: dimOut x 1 column Vector
	@Override
	public DoubleMatrix[] getOutput() {
		if (input != null && W != null) {
			DoubleMatrix[] output = new DoubleMatrix[1];
			output[0] = W.mmul(WeightUtil.flat2Vec(input));
			output[0].add(bias);
			return activate(output);
		}
		return null;
	}
	
	@Override
	public DoubleMatrix[] deriveDelta(DoubleMatrix[] outputDelta) {
		double[] input = outputDelta[0].transpose().mmul(W).toArray();
		DoubleMatrix[] inputDelta = new DoubleMatrix[numChannels];
		
		int index = 0;
		for (int i = 0; i < numChannels; i++) {
			inputDelta[i] = new DoubleMatrix(dimRows, dimCols);
			for (int n = 0; n < dimCols; n++) {
				for (int m = 0; m < dimRows; m++) {
					inputDelta[i].put(m, n, input[index++]);
				}
			}
		}
		return inputDelta;
	}

	@Override
	public DoubleMatrix[] update(DoubleMatrix[] outputDelta) {
		DoubleMatrix deltaW = outputDelta[0].mmul(WeightUtil.flat2Vec(input).transpose());
		W.addi(deltaW.mul(learningRate));
		
		return deriveDelta(outputDelta);
	}

	@Override
	public void initWeights() {
		W = WeightUtil.randInitWeights(dimOut, dimIn);
	}

	@Override
	public void applyDropOut() {
		// TODO Auto-generated method stub
	}

	

	
}
