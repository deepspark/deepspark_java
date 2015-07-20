package org.acl.deepspark.data;

import java.io.Serializable;

import org.jblas.DoubleMatrix;
import org.nd4j.linalg.api.ndarray.INDArray;

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
		for(int i = 0; i < gradWList.length; i++)
			result[i] = gradWList[i].div(num);
		return gradWList;
	}

	public void clear() {
		for(Weight weight : gradWList)
			weight = null;
		num = 0;
	}
}
