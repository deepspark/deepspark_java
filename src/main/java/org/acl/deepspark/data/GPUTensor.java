package org.acl.deepspark.data;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas;
import org.jblas.DoubleMatrix;

/**
 * Created by Jaehee on 2015-09-15.
 */
public class GPUTensor extends Tensor {
    // TODO
//    public Tensor mmul(Tensor t, char flag) {
//        assertMultipliesWith(t);
//        Tensor tensor = new Tensor(dimShape[0], dimShape[1], dimShape[2], t.dimShape[3]);
//
//        if(flag == 'c') {
//            for (int i = 0; i < tensor.dimShape[0]; i++) {
//                for (int j = 0; j < tensor.dimShape[1]; j++) {
//                    tensor.data[i][j] = data[i][j].mmul(t.data[i][j]);
//                }
//            }
//        } else {
//            for (int i = 0; i < tensor.dimShape[0]; i++) {
//                for (int j = 0; j < tensor.dimShape[1]; j++) {
//                    tensor.data[i][j] = dgemmJCublas(data[i][j], t.data[i][j]);
//                }
//            }
//        }
//        return tensor;
//    }

    public static DoubleMatrix dgemmJCublas(DoubleMatrix A, DoubleMatrix B)
    {
        int m = A.rows;
        int k = A.columns;
        int n = B.columns;

        int aa = (m > k)? m * m : k * k;
        int bb = (k > n)? k * k : n * n;
        int cc = (n > m)? n * n : m * m;

        DoubleMatrix C = DoubleMatrix.zeros(m, n);

        JCublas.cublasInit();

        Pointer d_A = new Pointer();
        Pointer d_B = new Pointer();
        Pointer d_C = new Pointer();
        JCublas.cublasAlloc(m * k, Sizeof.DOUBLE, d_A);
        JCublas.cublasAlloc(k * n, Sizeof.DOUBLE, d_B);
        JCublas.cublasAlloc(n * m, Sizeof.DOUBLE, d_C);

        JCublas.cublasSetVector(m * k, Sizeof.DOUBLE, Pointer.to(A.data), 1, d_A, 1);
        JCublas.cublasSetVector(k * n, Sizeof.DOUBLE, Pointer.to(B.data), 1, d_B, 1);
        JCublas.cublasSetVector(n * m, Sizeof.DOUBLE, Pointer.to(C.data), 1, d_C, 1);

        JCublas.cublasDgemm('n', 'n', m, n, k, 1, d_A, m, d_B, k, 0, d_C, m);

        JCublas.cublasGetVector(n * m, Sizeof.DOUBLE, d_C, 1, Pointer.to(C.data), 1);

        JCublas.cublasFree(d_A);
        JCublas.cublasFree(d_B);
        JCublas.cublasFree(d_C);

        JCublas.cublasShutdown();

        return C;
    } // TODO migrate to GPU

}
