package org.acl.deepspark.data;

import java.io.Serializable;

public class Weight implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = -2016361466768395491L;
	
	public Tensor w;
	public Tensor b;

	public Weight() {

	}

	public Weight(int weight[], int bias[]) {
		// Weight initialization, weight: gaussian, bias: zero
		w = Tensor.zeros(weight);
		b = Tensor.zeros(bias);
	}

	public Weight(Tensor w, Tensor b) {
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
		result.b = this.b.mul(d);
		return result;
	}

	public Weight muli(double d) {
		w.muli(d);
		b.muli(d);
		return this;
	}

	public Weight div(double d) {
		Weight result = new Weight();
		result.w = w.div(d);
		result.b = b.div(d);
		return result;
	}

	public Weight divi(double d) {
		w.divi(d);
		b.divi(d);
		return this;
	}

	public Weight dup() {
		return new Weight(w.dup(), b.dup());
	}

	public String toString() {
		return String.format("weight:\n%s\nbias:\n%s", w.toString(), b.toString());
	}

}
