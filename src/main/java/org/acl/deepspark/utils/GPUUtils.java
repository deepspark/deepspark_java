package org.acl.deepspark.utils;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas;
import org.acl.deepspark.data.Tensor;
import org.jblas.FloatMatrix;

/**
 * Created by Jaehee on 2015-09-15.
 */
public class GPUUtils extends Tensor {
    private static Pointer temp_A, temp_B, temp_C;
    private static final int GPU_BUFF_SIZE = 25 * 1024 * 1024;

    public static void preAllocationMemory() {
        temp_A = new Pointer();
        temp_B = new Pointer();
        temp_C = new Pointer();

        JCublas.cublasAlloc(GPU_BUFF_SIZE, Sizeof.FLOAT, temp_A);
        JCublas.cublasAlloc(GPU_BUFF_SIZE, Sizeof.FLOAT, temp_B);
        JCublas.cublasAlloc(GPU_BUFF_SIZE, Sizeof.FLOAT, temp_C);

    }

    public static void clearGPUMem() {
        JCublas.cublasFree(temp_A);
        JCublas.cublasFree(temp_B);
        JCublas.cublasFree(temp_C);
    }



    // TODO implementing elementwise operation

    /**
     * C = alpha * op(A) * op(B) + beta * op(C)
     * op(A) = A or A.transpose()
     * @param transa 'n' or 'N': op(A) = A / 't', 'T', 'c' or 'C': op(A) = A.transpose();
     * @param transb 'n' or 'N': op(B) = B / 't', 'T', 'c' or 'C': op(B) = B.transpose();
     */
    public static void sgemmJCublas(char transa, char transb, float alpha, FloatMatrix A, FloatMatrix B,
                                     float beta, FloatMatrix C) {
        int m = A.rows;
        int k = A.columns;
        int n = B.columns;

        JCublas.cublasInit();

        Pointer d_A = new Pointer();
        Pointer d_B = new Pointer();
        Pointer d_C = new Pointer();
        JCublas.cublasAlloc(m * k, Sizeof.FLOAT, d_A);
        JCublas.cublasAlloc(k * n, Sizeof.FLOAT, d_B);
        JCublas.cublasAlloc(n * m, Sizeof.FLOAT, d_C);

        JCublas.cublasSetVector(m * k, Sizeof.FLOAT, Pointer.to(A.data), 1, d_A, 1);
        JCublas.cublasSetVector(k * n, Sizeof.FLOAT, Pointer.to(B.data), 1, d_B, 1);
        JCublas.cublasSetVector(n * m, Sizeof.FLOAT, Pointer.to(C.data), 1, d_C, 1);

        JCublas.cublasDgemm(transa, transb, m, n, k, alpha, d_A, m, d_B, k, beta, d_C, m);

        JCublas.cublasGetVector(n * m, Sizeof.FLOAT, d_C, 1, Pointer.to(C.data), 1);

        JCublas.cublasFree(d_A);
        JCublas.cublasFree(d_B);
        JCublas.cublasFree(d_C);

        JCublas.cublasShutdown();
    }
}
