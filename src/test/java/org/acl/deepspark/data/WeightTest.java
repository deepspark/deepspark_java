package org.acl.deepspark.data;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by Jaehong on 2015-07-29.
 */
public class WeightTest {

    public static void main(String[] args) {
        double[] data1 = new double[] {1, -1, 1, 5, 1, 4, -8, 3, 10, 7, 8, 2,
                -5, 8, 3, 4, 1, -6, 23, 3, 3, 4, 2, 1,
                4, 12, 2, 2, 15, 50, 7, 23, 1, 2, -2, -1};

        double[] data2 = new double[] {1, 2, 1, 5, 1, 4, 8, 3, 2, 7, 9, 3,
                5, 8, 3, 4, 1, 12, 23, 34, 1, 4, 2, 1,
                4, 5, 23, 2, 1, 5, 7, 23, 1, 2, 4, 7};


        double[] bias1 = new double[] {1, 3};
        double[] bias2 = new double[] {2, 4};

        int[] dimIn = new int[] {4, 3, 3};
        INDArray input1 = Nd4j.create(data1, dimIn);
        INDArray input2 = Nd4j.create(data2, dimIn);

        Weight weight1 = new Weight(input1, Nd4j.create(bias1));
        Weight weight2 = new Weight(input2, Nd4j.create(bias2));

        System.out.println(weight1.toString());
        System.out.println(weight2.toString());
        System.out.println(weight1.mul(-1.3).toString());
        System.out.println(weight1.div(0.1).toString());

        weight1.addi(weight2);

        System.out.println(weight1);
        System.out.println(weight1.div(0.1).toString());


        
    }
}
