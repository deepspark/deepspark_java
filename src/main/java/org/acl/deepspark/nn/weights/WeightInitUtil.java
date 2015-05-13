
package org.acl.deepspark.nn.weights;

import org.jblas.DoubleMatrix;


public class WeightInitUtil {
	
	
	public static DoubleMatrix randInitWeights(int dimRow, int dimCol) {
		return DoubleMatrix.rand(dimRow, dimCol);
	}
}
