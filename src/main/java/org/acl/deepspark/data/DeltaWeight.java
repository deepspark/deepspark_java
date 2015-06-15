package org.acl.deepspark.data;

import org.jblas.DoubleMatrix;

public class DeltaWeight {
	public DoubleMatrix[][][] gradWList;
	public double[][] gradBList;
	
	public DeltaWeight(int numLayer) {
		gradWList = new DoubleMatrix[numLayer][][];
		gradBList = new double[numLayer][];
	}
}
