package org.acl.deepspark.nn.functions;

import org.acl.deepspark.data.Tensor;
import org.jblas.DoubleMatrix;

/**
 * Created by Jaehong on 2015-09-07.
 */
public class ActivatorTest {
    public static void main(String[] args) {
        double[] data = new double[] {1, 2, 1, 5, 1, 4, -8, 3, 2, 7, 9, 3,
                5, 8, 3, 4, -1, 12, -23, 34, 1, 4, 2, -1,
                4, 5, -23, 2, 1, 5, 7, 23, 1, -2, 4, 7};

        int[] dimIn = new int[] {4, 3, 3};
        Tensor input = Tensor.create(data, dimIn);
        System.out.println("input");
        System.out.println(input);

        Activator relu = ActivatorFactory.get(ActivatorType.RECTIFIED_LINEAR);
        Activator sigmoid = ActivatorFactory.get(ActivatorType.SIGMOID);
        Activator tanh = ActivatorFactory.get(ActivatorType.TANH);
        Activator none = ActivatorFactory.get(ActivatorType.NONE);

        System.out.println("relu");
        System.out.println(relu.output(input));
        System.out.println("sigmoid");
        System.out.println(sigmoid.output(input));
        System.out.println("tanh");
        System.out.println(tanh.output(input));
        System.out.println("none");
        System.out.println(none.output(input));
    }
}
