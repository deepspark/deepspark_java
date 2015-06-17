package org.acl.deepspark.data;

import java.io.Serializable;

import org.apache.spark.AccumulatorParam;
import org.jblas.DoubleMatrix;

public class DeltaAccumulator implements Serializable, AccumulatorParam<DeltaWeight> {

	/**
	 * 
	 */
	private static final long serialVersionUID = -5883736230420541223L;

	@Override
	public DeltaWeight addInPlace(DeltaWeight arg0, DeltaWeight arg1) {
		for(int i = 0; i < arg0.gradWList.length; i++) {
			if(arg0.gradWList[i] != null && arg1.gradWList[i] != null) {
				for(int j = 0; j < arg0.gradWList[i].length; j++) {
					for(int k = 0; k < arg0.gradWList[i][j].length;k++) {
						arg0.gradWList[i][j][k].addi(arg1.gradWList[i][j][k]);
					}
				}
				
				for(int j = 0; j < arg0.gradBList[i].length; j++) {
					arg0.gradBList[i][j] += arg1.gradBList[i][j];
				}	
			}
		}
		return arg0;
	}

	@Override
	public DeltaWeight zero(DeltaWeight arg0) {
		DeltaWeight d = new DeltaWeight(arg0.gradWList.length);
		for(int i = 0; i < arg0.gradWList.length; i++) {
			d.gradWList[i] = null;
			d.gradBList[i] = null;
			if(arg0.gradWList[i] != null) {
				d.gradWList[i] = new DoubleMatrix[arg0.gradWList[i].length][];
				for(int j = 0; j < arg0.gradWList[i].length; j++) {
					d.gradWList[i][j] = new DoubleMatrix[arg0.gradWList[i][j].length];
					for(int k = 0; k < arg0.gradWList[i][j].length;k++) {
						d.gradWList[i][j][k] = DoubleMatrix.zeros(arg0.gradWList[i][j][k].getRows(), arg0.gradWList[i][j][k].getColumns());
					}
				}
				
				d.gradBList[i] = new double[arg0.gradBList[i].length];
				for(int j = 0; j < arg0.gradBList[i].length; j++) {
					d.gradBList[i][j] = 0;
				}	
			}
		}
		
		return d;
	}

	@Override
	public DeltaWeight addAccumulator(DeltaWeight arg0, DeltaWeight arg1) {
		return addInPlace(arg0, arg1);
	}
}
