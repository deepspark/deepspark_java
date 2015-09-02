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

    private Tensor(int[] newDim) {
        this();
        if (newDim != null) {
            if (newDim.length > 4)
                throw new IllegalStateException(String.format("Only support (n <= 4) dimensional tensor, current: %d", newDim.length));
            /* dimShape = {kernels, channels, rows, cols} */
            System.arraycopy(newDim, 0, dimShape, 4-newDim.length, newDim.length);
            data = new DoubleMatrix[dimShape[0]][dimShape[1]];
        }
    }

    private Tensor(Tensor.init init, int[] newDim) {
        this(newDim);
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

    private Tensor(double[] newData, int[] newDim) {
        this(newDim);
        // TODO: assert sizeOf(newData) == sizeOf(newDim)

        double[] subArr = new double[dimShape[2] * dimShape[3]];
        for (int i = 0; i < dimShape[0]; i++) {
            for (int j = 0; j < dimShape[1]; j++) {
                int startPos = i * dimShape[1] * subArr.length + j * subArr.length;
                System.arraycopy(newData, startPos, subArr, 0, subArr.length);
                data[i][j] = new DoubleMatrix(dimShape[2], dimShape[3], subArr);
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

        Tensor tensor = new Tensor(dimShape);
        for (int i = 0 ; i < dimShape[0]; i++) {
            for (int j = 0; j < dimShape[1]; j++) {
                tensor.data[i][j] = data[i][j].add(t.data[i][j]);
            }
        }
        return tensor;
    }

    public Tensor addi(Tensor t) {
        assert Arrays.equals(dimShape, t.getShape());
        for (int i = 0 ; i < dimShape[0]; i++) {
            for (int j = 0; j < dimShape[1]; j++) {
                data[i][j].addi(t.data[i][j]);
            }
        }
        return this;
    }

    public Tensor sub(Tensor t) {
        assert Arrays.equals(dimShape, t.getShape());

        Tensor tensor = new Tensor(dimShape);
        for (int i = 0 ; i < dimShape[0]; i++) {
            for (int j = 0; j < dimShape[1]; j++) {
                tensor.data[i][j] = data[i][j].sub(t.data[i][j]);
            }
        }
        return tensor;
    }

    public Tensor subi(Tensor t) {
        assert Arrays.equals(dimShape, t.getShape());
        for (int i = 0 ; i < dimShape[0]; i++) {
            for (int j = 0; j < dimShape[1]; j++) {
                data[i][j].subi(t.data[i][j]);
            }
        }
        return this;
    }

    public Tensor mul(double d) {
        Tensor tensor = new Tensor(dimShape);
        for (int i = 0 ; i < dimShape[0]; i++) {
            for (int j = 0; j < dimShape[1]; j++) {
                tensor.data[i][j] = data[i][j].mul(d);
            }
        }
        return tensor;
    }

    public Tensor muli(double d) {
        for (int i = 0 ; i < dimShape[0]; i++) {
            for (int j = 0; j < dimShape[1]; j++) {
                data[i][j].muli(d);
            }
        }
        return this;
    }

    public Tensor div(double d) {
        Tensor tensor = new Tensor(dimShape);
        for (int i = 0 ; i < dimShape[0]; i++) {
            for (int j = 0; j < dimShape[1]; j++) {
                tensor.data[i][j] = data[i][j].div(d);
            }
        }
        return tensor;
    }

    public Tensor divi(double d) {
        for (int i = 0 ; i < dimShape[0]; i++) {
            for (int j = 0; j < dimShape[1]; j++) {
                data[i][j].divi(d);
            }
        }
        return this;
    }

    public Tensor dup() {
        Tensor tensor = new Tensor(dimShape.clone());
        for (int i = 0 ; i < dimShape[0]; i++) {
            for (int j = 0; j < dimShape[1]; j++) {
                tensor.data[i][j] = data[i][j].dup();
            }
        }
        return tensor;
    }

    public int[] getShape() {
        return dimShape;
    }

}
