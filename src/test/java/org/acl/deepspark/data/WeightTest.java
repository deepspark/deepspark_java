package org.acl.deepspark.data;

/**
 * Created by Jaehong on 2015-07-29.
 */
public class WeightTest {

    public static void main(String[] args) {
        float[] data1 = new float[] {1, -1, 1, 5, 1, 4, -8, 3, 10, 7, 8, 2,
                -5, 8, 3, 4, 1, -6, 23, 3, 3, 4, 2, 1,
                4, 12, 2, 2, 15, 50, 7, 23, 1, 2, -2, -1};

        float[] data2 = new float[] {1, 2, 1, 5, 1, 4, 8, 3, 2, 7, 9, 3,
                5, 8, 3, 4, 1, 12, 23, 34, 1, 4, 2, 1,
                4, 5, 23, 2, 1, 5, 7, 23, 1, 2, 4, 7};

        int[] dimIn = new int[] {4, 3, 3};
        float[] bias1 = new float[] {1, 3};
        float[] bias2 = new float[] {2, 4};

        Tensor input1 = Tensor.create(data1, dimIn);
        Tensor input2 = Tensor.create(data2, dimIn);

        Weight weight1 = new Weight(input1, Tensor.create(bias1, new int[]{2}));
        Weight weight2 = new Weight(input2, Tensor.create(bias2, new int[]{2}));

        System.out.println("weight1:" + weight1.toString());
        System.out.println("weight2:" + weight2.toString());

        System.out.println("weight1.mul(-1.3):\n" + weight1.mul((float) -1.3).toString());
        System.out.println("weight1.div(0.1):\n" + weight1.div((float) 0.1).toString());

        weight1.addi(weight2);
        System.out.println("weight1.addi(weight2):\n" + weight1.toString());

        weight1.subi(weight2);
        System.out.println("weight1.subi(weight2):\n" + weight1.toString());

        /** Weight Test complete **/
    }

}
