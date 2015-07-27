package org.acl.deepspark.data;

import java.io.Serializable;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Weight implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = -2016361466768395491L;
	
	public INDArray w;
	public INDArray b;

	public Weight() {

	}

	public Weight(int weight[], int bias[]) {
		// Weight initialization, weight: gaussian, bias: zero
		w = Nd4j.zeros(weight);
		b = Nd4j.zeros(bias);
	}

	public Weight(INDArray w, INDArray b) {
		this.w = w;
		this.b = b;
	}

	public int[] getWeightShape() {
		return w.shape();
	}

	public int[] getBiasShape() {
		return b.shape();
	}

	public Weight add(Weight weight) {
		Weight result = new Weight();
		result.w = this.w.add(weight.w);
		result.b = this.b.add(weight.b);
		return result;
	}

	public Weight addi(Weight weight) {
		w.addi(weight.w);
		b.addi(weight.b);
		return this;
	}

	public Weight sub(Weight weight) {
		Weight result = new Weight();
		result.w = this.w.sub(weight.w);
		result.b = this.b.sub(weight.b);
		return result;
	}

	public Weight subi(Weight weight) {
		w.subi(weight.w);
		b.subi(weight.b);
		return this;
	}

	public Weight mul(double d) {
		Weight result = new Weight();
		result.w = this.w.mul(d);
		result.b = this.b.sub(d);
		return result;
	}

	public Weight muli(double d) {
		w.muli(d);
		b.subi(d);
		return this;
	}

	public Weight div(double d) {
		Weight result = new Weight();
		result.w = this.w.div(d);
		result.b = this.b.div(d);
		return result;
	}

	public Weight divi(double d) {
		w.divi(d);
		b.divi(d);
		return this;
	}

	public void clear() {
		if (w != null && b != null) {

		}
	}

	// Matrix multiplication for weight x input
	public static INDArray mmul(Weight weight, INDArray input) {
		INDArray result = weight.w.mmul(input);
		return result.addi(weight.b);
	}

	// Matrix multiplication for input x weight
	public static INDArray mmul(INDArray input, Weight weight) {
		INDArray result = input.mmul(weight.w);
		return result.addi(weight.b);
	}


}
