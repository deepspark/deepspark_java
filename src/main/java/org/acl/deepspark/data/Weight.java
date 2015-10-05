package org.acl.deepspark.data;

import java.io.Serializable;

public class Weight implements Serializable {

	public Tensor w;
	public Tensor b;

	public static final float DEFAULT_VALUE = 0.0f;
	public static final WeightType DEFAULT_TYPE = WeightType.CONSTANT;

	private static final long serialVersionUID = -2016361466768395491L;

	public Weight() {
		w = null;
		b = null;
	}

	public Weight(int[] dimW, int[] dimB) {
		this.w = WeightFactory.create(DEFAULT_TYPE, DEFAULT_VALUE, dimW);
		this.b = WeightFactory.create(DEFAULT_TYPE, DEFAULT_VALUE, dimB);
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

	public Weight mul(float d) {
		Weight result = new Weight();
		result.w = this.w.mul(d);
		result.b = this.b.mul(d);
		return result;
	}

	public Weight muli(float d) {
		w.muli(d);
		b.muli(d);
		return this;
	}

	public Weight div(float d) {
		Weight result = new Weight();
		result.w = w.div(d);
		result.b = b.div(d);
		return result;
	}

	public Weight divi(float d) {
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
