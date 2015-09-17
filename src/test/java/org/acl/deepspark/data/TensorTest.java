package org.acl.deepspark.data;

/**
 * Created by Jaehong on 2015-09-02.
 */
public class TensorTest {
    public static void main(String[] args) {

        float[] data1 = new float[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        int[] dim1 = new int[] {3, 4};

        float[] data11 = new float[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        int[] dim11 = new int[] {2, 5};

        float[] data2 = new float[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                       13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        int[] dim2 = new int[] {2, 3, 4};

        float[] data22 = new float[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};

        int[] dim22 = new int[] {2, 3, 4};

        float[] data3 = new float[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                        25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
                                        37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48};
        int[] dim3 = new int[] {2, 2, 3, 4};

        float[] data33 = new float[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                        25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
                                        37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48};
        int[] dim33 = new int[] {2, 2, 3, 4};

        float[] data4 = new float[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
                37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48};
        int[] dim4 = new int[] {2, 2, 4, 3};

        float[] data44 = new float[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
                37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48};
        int[] dim44 = new int[] {2, 2, 4, 3};

        // create data
        System.out.println("create data");
        Tensor t1 = Tensor.create(data1, dim1);
        Tensor t11 = Tensor.create(data11, dim11);
        System.out.println(t1);
        Tensor t2 = Tensor.create(data2, dim2);
        Tensor t22 = Tensor.create(data22, dim22);
        System.out.println(t2);
        Tensor t3 = Tensor.create(data3, dim3);
        Tensor t33 = Tensor.create(data33, dim33);
        System.out.println(t3);
        Tensor t4 = Tensor.create(data4, dim4);
        Tensor t44 = Tensor.create(data44, dim44);
        System.out.println(t4);
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
        t3 = Tensor.create(data3, dim3);
        System.out.println("add");
        ret = t3.add(t3);
        System.out.println(ret);

        System.out.println("sub");
        ret = t3.sub(t3);
        System.out.println(ret);

        System.out.println("mul");
        ret = t3.mul(2);
        System.out.println(ret);

        System.out.println("mul");
//        ret = t1.mul(t11);
//        System.out.println(ret);
        ret = t3.mul(t33);
        System.out.println(ret);

        System.out.println("mmul");
//        ret = t3.mmul(t11);
//        System.out.println(ret);
//        ret = t3.mmul(t33);
//        System.out.println(ret);
        ret = t3.mmul(t4);
        System.out.println(ret);

        System.out.println("div");
        ret = t3.div(2);
        System.out.println(ret);

        System.out.println("div");
//        ret = t1.div(t11);
//        System.out.println(ret);
        ret = t3.div(t33);
        System.out.println(ret);

        // test complete

        // Matrix Operation (In place)
        Tensor tensor = Tensor.create(data3, dim3);
        System.out.println("addi");
        tensor.addi(Tensor.ones(dim3));
        System.out.println(tensor);

        System.out.println("subi");
        tensor.subi(Tensor.ones(dim3));
        System.out.println(tensor);

        System.out.println("muli");
        tensor.muli(2);
        System.out.println(tensor);

        System.out.println("muli");
//        tensor.muli(t1);
//        System.out.println(tensor);
        tensor.muli(t3);
        System.out.println(tensor);

        System.out.println("divi");
        tensor.divi(2);
        System.out.println(tensor);

        System.out.println("divi");
 //       tensor.muli(t1);
 //       System.out.println(tensor);
        tensor.divi(t3);
        System.out.println(tensor);
        // test complete

        // Sum
        System.out.println("sum");
        System.out.println(t3.sum());
        // test complete

        // Duplicate
        System.out.println("dup");
        Tensor tensor1 = tensor.dup();
        System.out.println(tensor);
        System.out.println(tensor1);
        System.out.println(tensor.equals(tensor1));
        // test complete

        // Transpose
        System.out.println("transpose");
        System.out.println(tensor1);
        tensor1 = tensor1.transpose();
        System.out.println(tensor1);
        tensor1 = tensor1.transpose();
        // test complete

        // Reshape
        System.out.println("reshape");
        tensor1 = tensor1.reshape(48);
        System.out.println(tensor1);
        // test complete

        t3 = Tensor.create(data3, dim3);
        t33 = Tensor.create(data33, dim33);

        // Merge
        System.out.println("merge");
//        ret = Tensor.merge(t3, t2);
        ret = Tensor.merge(t3, t33);
        System.out.println(ret);
        System.out.println(ret.equals(t3));
        System.out.println(ret.equals(t33));
        // test complete
    }
}
