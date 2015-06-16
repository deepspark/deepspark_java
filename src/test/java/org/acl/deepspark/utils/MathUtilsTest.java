package org.acl.deepspark.utils;

import org.jblas.DoubleMatrix;
import org.nd4j.linalg.convolution.Convolution;

public class MathUtilsTest {
	public static void main(String[] args) {
		double[][] a = {{1,3,5,7,9,11}, {13,11,9,7,5,3}, {10,6,8,4,2,1}, {9,7,5,3,1,3}, {14,12,10,8,6,4}, {16,14,7,9,8,3}};
		double[][] b = {{1,2,4,3,-1,0}, {2,4,3,5,7,-3}, {-2,1,-4,2,0,-3}, {0,-1,-2,2,3,-4}, {1,2,-3,-2,1,-1}, {3,2,1,-1,-2,3}};
		double[][] filterArr = {{2,3,5}, {1,11,3}, {10,6,4}};
		
		DoubleMatrix input1 = new DoubleMatrix(a);
		DoubleMatrix input2 = new DoubleMatrix(b);
		DoubleMatrix filter = new DoubleMatrix(filterArr);
		
		DoubleMatrix validConv1 = MathUtils.convolution(input1, filter, MathUtils.VALID_CONV);
		DoubleMatrix fullConv1  = MathUtils.convolution(input1, filter, MathUtils.FULL_CONV);
		
		DoubleMatrix validConv2 = MathUtils.convolution(input2, filter, MathUtils.VALID_CONV);
		DoubleMatrix fullConv2  = MathUtils.convolution(input2, filter, MathUtils.FULL_CONV);
		
		System.out.println("input1");
		System.out.println(input1);
		
		System.out.println("input2");
		System.out.println(input2);
		
		System.out.println("filter");
		System.out.println(filter);
		
		System.out.println("validConv1");
		System.out.println(validConv1);
		
		System.out.println("fullConv1");
		System.out.println(fullConv1);
		
		System.out.println("validConv2");
		System.out.println(validConv2);
		
		System.out.println("fullConv2");
		System.out.println(fullConv2);

		MathUtils.flip(filter);
		System.out.println("filter_flipped");
		System.out.println(filter.toString());
		
		/** Convolution Test Complete **/
	}
}
