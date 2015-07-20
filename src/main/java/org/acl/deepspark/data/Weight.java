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

	public Weight(int shape[]) {

	}

	public int[] getShape() {
		return w.shape();
	}

	public Weight add(Weight weight) {

	}

	public Weight addi(Weight weight) {

	}

	public Weight sub(Weight weight) {

	}

	public Weight subi(Weight weight) {

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
