package org.acl.deepspark.data;

import java.io.Serializable;

public class Accumulator implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = -948972344668801995L;
	
	public Weight[] gradWList;
	private int count;
	
	public Accumulator(int numLayer) {
		gradWList = new Weight[numLayer];
		count = 0;
	}

	public void accumulate(Weight[] weights) {
		if (gradWList.length != weights.length)
			throw new IllegalArgumentException(String.format
					("Number of layers mismatch; current %d, param %d", gradWList.length, weights.length));

		for (int i = 0 ; i < gradWList.length; i++) {
			if (gradWList[i] == null) {
				if (weights[i] != null)
					gradWList[i] = weights[i].dup();
			}
			else
				gradWList[i].addi(weights[i]);
		}
		count++;
	}

	public int getCount() {
		return count;
	}

	public Weight[] getAverage() {
		if (count <= 0) return null;

		Weight[] result = new Weight[gradWList.length];
		for (int i = 0; i < gradWList.length; i++)
			if (gradWList[i] != null)
				result[i] = gradWList[i].div(count);
		return result;
	}

	public void clear() {
		for (int i = 0; i < gradWList.length; i++)
			gradWList[i] = null;
		count = 0;
	}
}
