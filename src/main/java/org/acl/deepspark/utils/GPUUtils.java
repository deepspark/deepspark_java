package org.acl.deepspark.utils;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas;
import org.jblas.FloatMatrix;
import org.jblas.util.Random;

/**
 * Created by Jaehee on 2015-09-15.
 */
public class GPUUtils {
    private static Pointer temp_A, temp_B, temp_C;
    private static final int GPU_BUFF_SIZE = 25 * 1024 * 1024;

    public static void init() {
        JCublas.cublasInit();
        preAllocationMemory();
    }

    public static void shutdown() {
        clearGPUMem();
        JCublas.cublasShutdown();
    }

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

    /**
     * C = alpha * op(A) * op(B) + beta * op(C)
     * op(A) = A or A.transpose()
     * @param transa 'n' or 'N': op(A) = A / 't', 'T', 'c' or 'C': op(A) = A.transpose();
     * @param transb 'n' or 'N': op(B) = B / 't', 'T', 'c' or 'C': op(B) = B.transpose();
     */
    public static void sgemmJCublas(char transa, char transb, float alpha, FloatMatrix A, FloatMatrix B,
                                     float beta, FloatMatrix C) {
        int m = (transa == 'n') ? A.rows : A.columns;
        int k = (transa == 'n') ? A.columns : A.rows;
        int n = (transb == 'n') ? B.columns : B.rows;

        JCublas.cublasSetVector(m * k, Sizeof.FLOAT, Pointer.to(A.data), 1, temp_A, 1);
        JCublas.cublasSetVector(k * n, Sizeof.FLOAT, Pointer.to(B.data), 1, temp_B, 1);
        JCublas.cublasSetVector(n * m, Sizeof.FLOAT, Pointer.to(C.data), 1, temp_C, 1);

        JCublas.cublasSgemm(transa, transb, m, n, k, alpha, temp_A, m, temp_B, k, beta, temp_C, m);

        JCublas.cublasGetVector(n * m, Sizeof.FLOAT, temp_C, 1, Pointer.to(C.data), 1);
    }

    /**
     * B = alpha * A + B
     *
     * @param alpha
     * @param A
     * @param B
     */
    private static void saxpyJCublas(float alpha, FloatMatrix A, FloatMatrix B) {
        int n = A.rows * A.columns;

        JCublas.cublasSetVector(n, Sizeof.FLOAT, Pointer.to(A.data), 1, temp_A, 1);
        JCublas.cublasSetVector(n, Sizeof.FLOAT, Pointer.to(B.data), 1, temp_B, 1);

        JCublas.cublasSaxpy(n, alpha, temp_A, 1, temp_B, 1);

        JCublas.cublasGetVector(n, Sizeof.FLOAT, temp_B, 1, Pointer.to(B.data), 1);

        // JCublas.cublasShutdown();
    }

    public static FloatMatrix randomFloatMatrix(int n, int m) {
        int size = n * m;
        float[] input = new float[size];
        for(int i = 0; i < size; i++) {
            input[i] = Random.nextFloat();
        }
        FloatMatrix result = new FloatMatrix(n, m, input);
        return result;
    }

    public static void main(String[] args) {
        int n = 4096;
        GPUUtils.preAllocationMemory();
        double start = System.currentTimeMillis();

        for(int i = 0; i < 100; i++) {
            FloatMatrix A = GPUUtils.randomFloatMatrix(n, n);
            FloatMatrix B = GPUUtils.randomFloatMatrix(n, n);
            GPUUtils.saxpyJCublas(1, A, B);
        }

        double end = System.currentTimeMillis();

        System.out.println("gpu: "+(end - start) / 1000);

        start = System.currentTimeMillis();

        for(int i = 0; i < 100; i++) {
            FloatMatrix A = GPUUtils.randomFloatMatrix(n, n);
            FloatMatrix B = GPUUtils.randomFloatMatrix(n, n);
            B.addi(A);
        }

        end = System.currentTimeMillis();

        System.out.println("cpu: "+(end - start) / 1000);
        GPUUtils.shutdown();
    }
}
