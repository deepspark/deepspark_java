package org.acl.deepspark.data;

import org.apache.commons.math.DimensionMismatchException;

import java.io.Serializable;

public class Accumulator implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = -948972344668801995L;
	
	public Weight[] gradWList;
	private int num;
	
	public Accumulator(int numLayer) {
		gradWList = new Weight[numLayer];
		num = 0;
	}

	public void accumulate(Weight[] weights) {
		if (gradWList.length != weights.length)
			throw new IllegalArgumentException("Only rotating matrices");

		for (int i = 0 ; i < gradWList.length; i++) {
			if (gradWList[i] == null) {
				if (weights[i] != null)
					gradWList[i] = weights[i].dup();
			}
			else
				gradWList[i].addi(weights[i]);
		}
		num++;
	}

	public Weight[] getAverage() {
		if (num <= 0) return null;

		Weight[] result = new Weight[gradWList.length];
		for (int i = 0; i < gradWList.length; i++)
			if (gradWList[i] != null)
				result[i] = gradWList[i].div(num);
		return result;
	}

	public void clear() {
		for (int i = 0; i < gradWList.length; i++)
			gradWList[i] = null;
		num = 0;
	}
}
