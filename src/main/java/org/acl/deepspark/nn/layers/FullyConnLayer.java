package org.acl.deepspark.nn.layers;

import java.io.Serializable;

import org.acl.deepspark.nn.conf.LayerConf;
import org.acl.deepspark.nn.functions.Activator;
import org.acl.deepspark.nn.weights.WeightUtil;
import org.jblas.DoubleMatrix;
import org.nd4j.linalg.api.ndarray.INDArray;

// Fully Connected HiddenLayer
public class FullyConnLayer extends BaseLayer implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = -7250079452692054259L;
	private DoubleMatrix W;
	private DoubleMatrix output;
	private double momentumFactor = 0.0;
	private DoubleMatrix prevDeltaW;
	private double prevDeltaBias;
	private double bias = 0.01;
	private double decayLambda = 0.00001;

	public FullyConnLayer() {

	}

	public FullyConnLayer(int nOut) {
		this.dimOut = nOut;
	}
	
	public FullyConnLayer(int nOut,double momentum, double decayLambda) {
		super(Activator.SOFTMAX);
		this.dimOut = nOut;
		this.momentumFactor = momentum;
		this.decayLambda = decayLambda;
	}
	
	public FullyConnLayer(int nOut,double momentum, double decayLambda, int activator) {
		super(activator);
		this.dimOut = nOut;
		this.momentumFactor = momentum;
		this.decayLambda = decayLambda;
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
	public DoubleMatrix[] deriveDelta() {
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
	public void setDelta(DoubleMatrix[] delta) {
		this.delta = delta;
	}
	
	@Override
	public void update(DoubleMatrix[][] gradW, double[] gradB) {
		prevDeltaW.muli(momentumFactor);
		prevDeltaW.addi(W.mul(learningRate * 2* decayLambda ));
		prevDeltaW.addi(gradW[0][0].muli(learningRate));
		
		//prevDeltaBias *= momentumFactor;
		prevDeltaBias = bias * decayLambda * learningRate;
		prevDeltaBias += gradB[0] * learningRate;
		 
		// weight update
		W.subi(prevDeltaW);
		bias -= prevDeltaBias;
		// propagate delta to the previous layer
	}

	@Override
	public void initWeights() {
		if(W == null) {
			W = WeightUtil.randInitWeights(dimOut, dimIn, dimIn);
			bias = 0.01;
			prevDeltaW = DoubleMatrix.zeros(dimOut, dimIn);
			prevDeltaBias = 0;
		}
	}

	@Override
	public void applyDropOut() {
		// TODO Auto-generated method stub
	}

	@Override
	public DoubleMatrix[][] deriveGradientW() {
		DoubleMatrix[][] grad = new DoubleMatrix[1][1];
		grad[0][0] = delta[0].mmul(WeightUtil.flat2Vec(input).transpose());
		return grad;
	}

	@Override
	public int[] initWeights(int[] dim) {
		int[] outDim = new int[1];
		int inDim = 1;
		for(int i = 0; i < dim.length; i++) {
			inDim *= dim[i];
		}
		dimIn = inDim;
		outDim[0] = dimOut;
		
		initWeights();
		return outDim;
	}

	@Override
	public int[] getWeightInfo() {
		int[] info = {1, 1, dimOut, dimIn};
		return info;
	}

	@Override
	public INDArray createWeight(LayerConf conf, int[] input) {
		return null;
	}

	@Override
	public INDArray generateOutput(INDArray weight, INDArray input) {
		return null;
	}

	@Override
	public INDArray deriveDelta(INDArray weight, INDArray error) {
		return null;
	}

	@Override
	public INDArray gradient(INDArray input, INDArray error) {
		return null;
	}
}
