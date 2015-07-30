package org.acl.deepspark.utils;

import org.jblas.util.Random;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by Jaehong on 2015-07-29.
 */
public class ArrayUtilsTest {
    public static void main(String[] args) {
        double[] input_a = new double[]{1, 2, -2, 4, -2, 6, 3, -1, 11};
        INDArray a = Nd4j.create(input_a, new int[]{3, 3});
        double[] input_b = new double[]{-1, 6, 2, 8, 1, 1, 1, 2, 0};
        INDArray b = Nd4j.create(input_b, new int[]{3, 3});
        double[] input_c = new double[]{1, 0, -2, 3};
        INDArray c = Nd4j.create(input_c, new int[]{2, 2});

        INDArray valid1 = ArrayUtils.convolution(a, b, ArrayUtils.VALID_CONV);
        INDArray full1 = ArrayUtils.convolution(a, b, ArrayUtils.FULL_CONV);

        INDArray valid2 = ArrayUtils.convolution(a, c, ArrayUtils.VALID_CONV);
        INDArray full2 = ArrayUtils.convolution(a, c, ArrayUtils.FULL_CONV);

        System.out.println("input_a: " + a);
        System.out.println("input_b: " + b);
        System.out.println("valid: " + valid1);
        System.out.println("full: " + full1);

        System.out.println("input_a: " + a);
        System.out.println("input_c: " + c);
        System.out.println("valid: " + valid2);
        System.out.println("full: " + full2);
    }
}
