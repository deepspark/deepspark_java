package org.acl.deepspark.data;

import org.jblas.DoubleMatrix;

import java.io.Serializable;
import java.util.Arrays;

/**
 * Created by Jaehong on 2015-09-02.
 */
public class Tensor implements Serializable {

    private int[] dimShape;         // dimShape = {kernels, channels, rows, cols}
    private DoubleMatrix[][] data;  // data = DoubleMatrix[kernels][channels]

    public enum init {
        ZEROS, UNIFORM, GAUSSIAN
    }

    private Tensor() {
        dimShape = new int[] {1, 1, 1, 1};
    }

    public Tensor(Tensor.init init, int[] newDim) {
        this();
        if (newDim != null) {
            if (newDim.length > 4)
                throw new IllegalStateException(String.format("Only support (n <= 4) dimensional tensor, current: %d", newDim.length));

            /* dimShape = {kernels, channels, rows, cols} */
            System.arraycopy(newDim, 0, dimShape, 4-newDim.length, newDim.length);
            data = new DoubleMatrix[dimShape[0]][dimShape[1]];

            for (int i = 0; i < dimShape[0]; i++) {
                for (int j = 0; j < dimShape[1]; j++) {
                    switch (init) {
                        case ZEROS:
                            data[i][j] = DoubleMatrix.zeros(dimShape[2], dimShape[3]);
                            break;

                        case UNIFORM:
                            data[i][j] = DoubleMatrix.rand(dimShape[2], dimShape[3]);
                            break;

                        case GAUSSIAN:
                            data[i][j] = DoubleMatrix.randn(dimShape[2], dimShape[3]);
                            break;
                    }
                }
            }
        }
    }

    public Tensor(double[] newData, int[] newDim) {
        this();
        if (newDim != null) {
            if (newDim.length > 4)
                throw new IllegalStateException(String.format("Only support (n <= 4) dimensional tensor, current: %d", newDim.length));

            /* dimShape = {kernels, channels, rows, cols} */
            System.arraycopy(newDim, 0, dimShape, 0, newDim.length);
            data = new DoubleMatrix[dimShape[0]][dimShape[1]];

            double[] subArr = new double[dimShape[2]*dimShape[3]];
            for (int i = 0; i < dimShape[0]; i++) {
                for (int j = 0; j < dimShape[1]; j++) {
                    int startPos = i*dimShape[1]*subArr.length + j*subArr.length;
                    System.arraycopy(newData, startPos, subArr, 0, subArr.length);
                    data[i][j] = new DoubleMatrix(dimShape[2], dimShape[3], subArr);
                }
            }
        }
    }

    public static Tensor create(double[] newData, int[] newDim) {
        return new Tensor(newData, newDim);
    }

    public static Tensor zeros(int[] shape) {
        return new Tensor(init.ZEROS, shape);
    }

    public static Tensor rand(int[] shape) {
        return new Tensor(init.UNIFORM, shape);
    }

    public static Tensor randn(int[] shape) {
        return new Tensor(init.GAUSSIAN, shape);
    }

    public Tensor add(Tensor t) {
        assert Arrays.equals(dimShape, t.getShape());

        Tensor tensor = new Tensor(init.ZEROS, dimShape);

        for (int i = 0 ; i < dimShape[0]; i++) {
            for (int j = 0; j < dimShape[1]; j++) {
            }
        }
    }

    public Tensor addi(Tensor t) {

        return this;
    }

    public Tensor sub(Tensor t) {
        Weight result = new Weight();
        result.w = this.w.sub(weight.w);
        result.b = this.b.sub(weight.b);
        return result;
    }

    public Tensor subi(Tensor t) {
        w.subi(weight.w);
        b.subi(weight.b);
        return this;
    }

    public Tensor mul(double d) {
        Weight result = new Weight();
        result.w = this.w.mul(d);
        result.b = this.b.mul(d);
        return result;
    }

    public Tensor muli(double d) {
        w.muli(d);
        b.muli(d);
        return this;
    }

    public Tensor div(double d) {
        Weight result = new Weight();
        result.w = w.div(d);
        result.b = b.div(d);
        return result;
    }

    public Tensor divi(double d) {
        w.divi(d);
        b.divi(d);
        return this;
    }

    public Tensor dup() {
        return new Weight(w.dup(), b.dup());
    }

    public int[] getShape() {
        return dimShape;
    }
}
