
package org.acl.deepspark.nn.weights;

import org.jblas.DoubleMatrix;

import java.io.Serializable;


public class WeightUtil implements Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -1196601489934475280L;

	public static DoubleMatrix randInitWeights(int dimRow, int dimCol) {
		return randInitWeights(dimRow, dimCol, 2);
	}
	
	public static DoubleMatrix randInitWeights(int dimRow, int dimCol, int numInput) {
		return DoubleMatrix.randn(dimRow, dimCol).muli(Math.sqrt(2/numInput));
	}
	
	public static DoubleMatrix accum(DoubleMatrix[] matrices) {
		DoubleMatrix result = new DoubleMatrix(matrices[0].rows, matrices[0].columns);
		for(DoubleMatrix matrix : matrices) {
			result.add(matrix);
		}
		return result;
	}
	
	public static DoubleMatrix flat2Vec(DoubleMatrix matrix) {
		if (matrix != null)
			return new DoubleMatrix(matrix.toArray());
		return null;
	}
	
	public static DoubleMatrix flat2Vec(DoubleMatrix[] matrices) {
		if (matrices != null) {
			int length = matrices.length;
			int dim = matrices[0].length;
			
			double[] data = new double[length * dim];
			for(int i = 0 ; i < length; i++) {
				for (int j = 0; j < dim; j++) {
					data[i * dim + j] = matrices[i].get(j);
				}
			}
			return new DoubleMatrix(data);
		}
		return null;
	}
}
