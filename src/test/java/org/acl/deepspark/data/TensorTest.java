package org.acl.deepspark.data;

/**
 * Created by Jaehong on 2015-09-02.
 */
public class TensorTest {
    public static void main(String[] args) {

        double[] data1 = new double[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        int[] dim1 = new int[] {3, 4};

        double[] data2 = new double[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                       13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        int[] dim2 = new int[] {2, 3, 4};

        double[] data3 = new double[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                        25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
                                        37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48};
        int[] dim3 = new int[] {2, 2, 3, 4};

        // create data
        System.out.println("create data");
        Tensor t1 = Tensor.create(data1, dim1);
        System.out.println(t1);
        Tensor t2 = Tensor.create(data2, dim2);
        System.out.println(t2);
        Tensor t3 = Tensor.create(data3, dim3);
        System.out.println(t3);
        // test complete

        // create zeros
        System.out.println("create zeros");
        t1 = Tensor.zeros(dim1);
        System.out.println(t1);
        t2 = Tensor.zeros(dim2);
        System.out.println(t2);
        t3 = Tensor.zeros(dim3);
        System.out.println(t3);
        // test complete

        // create rand
        System.out.println("create rand");
        t1 = Tensor.rand(dim1);
        System.out.println(t1);
        t2 = Tensor.rand(dim2);
        System.out.println(t2);
        t3 = Tensor.rand(dim3);
        System.out.println(t3);
        // test complete

        // create randn
        System.out.println("create randn");
        t1 = Tensor.randn(dim1);
        System.out.println(t1);
        t2 = Tensor.randn(dim2);
        System.out.println(t2);
        t3 = Tensor.randn(dim3);
        System.out.println(t3);
        // test complete

        // Matrix Operation
        Tensor ret;
        System.out.println("add");
        ret = t2.add(t2);
        System.out.println(ret);

        System.out.println("sub");
        ret = t2.sub(t2);
        System.out.println(ret);

        System.out.println("mul");
        ret = t2.mul(2.0);
        System.out.println(ret);

        System.out.println("div");
        ret = t2.div(2.0);
        System.out.println(ret);
/*
        // Matrix Operation (In place)
        Tensor tensor = Tensor.create(data3, dim3);;
        System.out.println("addi");
        tensor.addi(Tensor.ones(dim3));
        System.out.println(tensor);

        System.out.println("subi");
        tensor.subi(Tensor.ones(dim3));
        System.out.println(tensor);

        System.out.println("muli");
        tensor.muli(2.0);
        System.out.println(tensor);

        System.out.println("divi");
        tensor.divi(2.0);
        System.out.println(tensor);
*/
    }
}
