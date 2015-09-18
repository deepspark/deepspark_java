package org.acl.deepspark.data;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas;
import org.jblas.FloatMatrix;

/**
 * Created by Jaehee on 2015-09-15.
 */
public class GPUTensor extends Tensor {
    // TODO

    public Tensor add(Tensor t) {
        assertSameLength(t);

        Tensor tensor = new Tensor(t.data, dimShape);
        for (int i = 0; i < data.length; i++) {
            // warnings will be gone once data is modified for FloatMatrix
            sgemmJCublas('n', 'n', 1, data[i], FloatMatrix.eye(data[i].columns), 1, tensor.data[i]);
        }

        return tensor;
    }

    public Tensor addi(Tensor t) {
        assertSameLength(t);
        FloatMatrix[] copy = new FloatMatrix[data.length];
        System.arraycopy(data, 0, copy, 0, data.length);
        for (int i = 0; i < data.length; i++) {
            sgemmJCublas('n', 'n', 1, copy[i], FloatMatrix.eye(data[i].columns), 1, data[i]); // TODO axpy?
        }

        return this;
    }

    public Tensor sub(Tensor t) {
        assertSameLength(t);
        Tensor tensor = new Tensor(t.data, dimShape);
        for (int i = 0; i < data.length; i++) {
            sgemmJCublas('n', 'n', 1, data[i], FloatMatrix.eye(data[i].columns), -1, tensor.data[i]);
        }
    }

    public Tensor subi(Tensor t) {
        assertSameLength(t);
        FloatMatrix[] copy = new FloatMatrix[data.length];
        System.arraycopy(data, 0, copy, 0, data.length);
        for(int i = 0; i < data.length; i++) {
            sgemmJCublas('n', 'n', copy[i], FloatMatrix.eye(data[i].columns), -1, data[i]);
        }

        return this;
    }

    public Tensor mmul(Tensor t) {
        assertMultipliesWith(t);
        Tensor tensor = new Tensor(dimShape[0], dimShape[1], dimShape[2], t.dimShape[3]);
        int length = data.length;
        for (int i = 0 ; i < length; i++) {
            sgemmJCublas('n', 'n', 1, data, t.data, 0, tensor.data);
        }
        return tensor;
    }

    // TODO implementing elementwise operation

    /**
     * C = alpha * op(A) * op(B) + beta * op(C)
     * op(A) = A or A.transpose()
     * @param transa 'n' or 'N': op(A) = A / 't', 'T', 'c' or 'C': op(A) = A.transpose();
     * @param transb 'n' or 'N': op(B) = B / 't', 'T', 'c' or 'C': op(B) = B.transpose();
     */
    private void sgemmJCublas(char transa, char transb, float alpha, FloatMatrix A, FloatMatrix B,
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
