package org.acl.deepspark.utils;

import org.acl.deepspark.data.Tensor;
import org.jblas.DoubleMatrix;

/**
 * Created by Jaehong on 2015-07-29.
 */
public class ArrayUtilsTest {
    public static void main(String[] args) {
        double[] input_a = new double[]{1, 2, -2, 4, -2, 6, 3, -1, 11};
        Tensor a = Tensor.create(input_a, new int[]{3, 3});
        double[] input_b = new double[]{-1, 6, 2, 8, 1, 1, 1, 2, 0};
        Tensor b = Tensor.create(input_b, new int[]{3, 3});
        double[] input_c = new double[]{1, 0, -2, 3};
        Tensor c = Tensor.create(input_c, new int[]{2, 2});

        DoubleMatrix valid1 = ArrayUtils.convolution(a.slice(0,0), b.slice(0,0), ArrayUtils.VALID_CONV);
        DoubleMatrix full1 = ArrayUtils.convolution(a.slice(0,0), b.slice(0,0), ArrayUtils.FULL_CONV);

        DoubleMatrix valid2 = ArrayUtils.convolution(a.slice(0,0), c.slice(0,0), ArrayUtils.VALID_CONV);
        DoubleMatrix full2 = ArrayUtils.convolution(a.slice(0,0), c.slice(0,0), ArrayUtils.FULL_CONV);

        /*
         * valid: 44.0
			full: [0.0, 1.0, 6.0, 11.0, 6.0]
			      [2.0, 11.0, 18.0, 22.0, 16.0]
			      [5.0, 8.0, 44.0, 30.0, 13.0]
			      [-2.0, 24.0, -3.0, 41.0, 67.0]
			      [-2.0, -10.0, 61.0, 82.0, -11.0]
         */
        
        System.out.println("input_a: " + a);
        System.out.println("input_b: " + b);
        System.out.println("valid: " + valid1);
        System.out.println("full: " + full1);

        /*
          	valid:  [-13.0, -5.0]
 				    [24.0, 33.0]]
        	full:   [3.0, 12.0, 9.0, 0.0]
        		    [4.0, -13.0, -5.0, 3.0]
        		    [-10.0, 24.0, 33.0, -1.0]
        		    [4.0, -14.0, -16.0, 11.0]
		 */
        System.out.println("input_a: " + a);
        System.out.println("input_c: " + c);
        System.out.println("valid: " + valid2);
        System.out.println("full: " + full2);

        /** convolution test complete **/


    }
}
