
package org.acl.deepspark.nn.weights;

import org.jblas.DoubleMatrix;


public class WeightUtil {
	
	
	public static DoubleMatrix randInitWeights(int dimRow, int dimCol) {
		return DoubleMatrix.rand(dimRow, dimCol);
	}
	
	public static DoubleMatrix accum(DoubleMatrix[] matrices) {
		DoubleMatrix result = new DoubleMatrix(matrices[0].rows, matrices[0].columns);
		for(DoubleMatrix matrix : matrices) {
			result.add(matrix);
		}
		return result;
	}
	
	public static DoubleMatrix flip(DoubleMatrix matrix) {
		return matrix;
	}
}
