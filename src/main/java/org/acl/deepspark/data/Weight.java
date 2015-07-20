package org.acl.deepspark.data;

import java.io.Serializable;

import org.nd4j.linalg.api.ndarray.INDArray;

public class Weight implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = -2016361466768395491L;
	
	public INDArray w;
	public INDArray b;

	public Weight() {

	}

	public void add(Weight weight) {

	}

	public void addi(Weight weight) {

	}

	public void sub(Weight weight) {

	}

	public void subi(Weight weight) {

	}

	public Weight mul(double d) {
		return null;
	}

	public Weight muli(double d) {
		return null;
	}

	public INDArray mmul(INDArray input) {
		return null;
	}
}
