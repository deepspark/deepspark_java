package org.acl.deepspark.data;

import java.io.Serializable;

public class Accumulator implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = -948972344668801995L;
	
	private Weight[] gradWList;
	private int num;
	
	public Accumulator(int numLayer) {
		gradWList = new Weight[numLayer];
		num = 0;
	}

	public void accumulate(Weight[] weights) {
		if (gradWList.length != weights.length)
			System.out.println("Weight dimension mismatch");
		//	throw new Exception("Weight dimension mismatch");
		for (int i = 0 ; i < gradWList.length; i++) {
			if (gradWList[i] == null)
				gradWList[i] = weights[i];
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
