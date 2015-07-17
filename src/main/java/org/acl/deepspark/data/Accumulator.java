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
	}

	public Weight getAverage() {
		for (INDArray weight : gradWList) {
			// TODO: check whether divi() performs well
			weight.divi(num);
		}

	}

	public void clear() {

	}
}
