package org.acl.deepspark.utils;

import org.acl.deepspark.data.Tensor;
import org.jblas.FloatMatrix;

/**
 * Created by Jaehong on 2015-07-29.
 */
public class ArrayUtilsTest {
    public static void main(String[] args) {
        float[] input_a = new float[]{1, 2, -2, 4, -2, 6, 3, -1, 11};
        Tensor a = Tensor.create(input_a, new int[]{3, 3});
        float[] input_b = new float[]{-1, 6, 2, 8, 1, 1, 1, 2, 0};
        Tensor b = Tensor.create(input_b, new int[]{3, 3});
        float[] input_c = new float[]{1, 0, -2, 3};
        Tensor c = Tensor.create(input_c, new int[]{2, 2});

        FloatMatrix valid1 = ArrayUtils.convolution(a.slice(0,0), b.slice(0,0), ArrayUtils.VALID_CONV);
        FloatMatrix full1 = ArrayUtils.convolution(a.slice(0,0), b.slice(0,0), ArrayUtils.FULL_CONV);

        FloatMatrix valid2 = ArrayUtils.convolution(a.slice(0,0), c.slice(0,0), ArrayUtils.VALID_CONV);
        FloatMatrix full2 = ArrayUtils.convolution(a.slice(0,0), c.slice(0,0), ArrayUtils.FULL_CONV);

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


        float[] data = new float[] {1, 2, 1, 5, 1, 4, 8, 3, 2, 7, 9, 3,
                5, 8, 3, 4, 1, 12, 23, 34, 1, 4, 2, 1,
                4, 5, 23, 2, 1, 5, 7, 23, 1, 2, 4, 7};

        int[] dimIn = new int[] {4, 3, 3};
        Tensor input = Tensor.create(data, dimIn);
        System.out.println("input");
        System.out.println(input);

        Tensor padInput = ArrayUtils.zeroPad(input, 2);
        System.out.println("padInput");
        System.out.println(padInput);

        Tensor cropped = ArrayUtils.centerCrop(padInput, 2);
        System.out.println("cropped");
        System.out.println(cropped);


    }
}
