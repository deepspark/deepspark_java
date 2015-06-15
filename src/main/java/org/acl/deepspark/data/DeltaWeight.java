package org.acl.deepspark.data;

import java.io.Serializable;

import org.jblas.DoubleMatrix;

public class DeltaWeight implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = -948972344668801995L;
	
	public DoubleMatrix[][][] gradWList;
	public double[][] gradBList;
	
	public DeltaWeight(int numLayer) {
		gradWList = new DoubleMatrix[numLayer][][];
		gradBList = new double[numLayer][];
	}
}
