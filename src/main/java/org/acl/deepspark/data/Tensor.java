package org.acl.deepspark.data;

import org.jblas.DoubleMatrix;
import org.jblas.exceptions.SizeException;

import java.io.Serializable;
import java.util.Arrays;

/**
 * Created by Jaehong on 2015-09-02.
 */
public class Tensor implements Serializable {

    private int[] dimShape;         // dimShape = {kernels, channels, rows, cols}
    private DoubleMatrix[][] data;  // data = DoubleMatrix[kernels][channels]

    public enum init {
        ZEROS, ONES, UNIFORM, GAUSSIAN
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

                    case ONES:
                        data[i][j] = DoubleMatrix.ones(dimShape[2], dimShape[3]);
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
        assertMatchSize(newData, newDim);

        for (int i = 0; i < dimShape[0]; i++) {
            for (int j = 0; j < dimShape[1]; j++) {
                double[] subArr = new double[dimShape[2] * dimShape[3]];
                int startPos = i * dimShape[1] * subArr.length + j * subArr.length;
                System.arraycopy(newData, startPos, subArr, 0, subArr.length);
                data[i][j] = new DoubleMatrix(dimShape[2], dimShape[3], subArr);
            }
        }
    }

    public int[] shape() {
        return dimShape;
    }

    public int length() {
        int length = 1;
        for (int dim : dimShape)
            length *= dim;
        return length;
    }

    public DoubleMatrix[] slice(int kernelIdx) {
        return data[kernelIdx];
    }

    public DoubleMatrix slice(int kernelIdx, int channelIdx) {
        return data[kernelIdx][channelIdx];
    }

    public static Tensor create(double[] newData, int[] newDim) {
        return new Tensor(newData, newDim);
    }

    public static Tensor zeros(int[] shape) {
        return new Tensor(init.ZEROS, shape);
    }

    public static Tensor ones(int[] shape) {
        return new Tensor(init.ONES, shape);
    }

    public static Tensor rand(int[] shape) {
        return new Tensor(init.UNIFORM, shape);
    }

    public static Tensor randn(int[] shape) {
        return new Tensor(init.GAUSSIAN, shape);
    }

    public Tensor add(Tensor t) {
        assertSameLength(t);

        Tensor tensor = new Tensor(dimShape);
        for (int i = 0 ; i < dimShape[0]; i++) {
            for (int j = 0; j < dimShape[1]; j++) {
                tensor.data[i][j] = data[i][j].add(t.data[i][j]);
            }
        }
        return tensor;
    }

    public Tensor addi(Tensor t) {
        assertSameLength(t);
        for (int i = 0 ; i < dimShape[0]; i++) {
            for (int j = 0; j < dimShape[1]; j++) {
                data[i][j].addi(t.data[i][j]);
            }
        }
        return this;
    }

    public Tensor sub(Tensor t) {
        assertSameLength(t);

        Tensor tensor = new Tensor(dimShape);
        for (int i = 0 ; i < dimShape[0]; i++) {
            for (int j = 0; j < dimShape[1]; j++) {
                tensor.data[i][j] = data[i][j].sub(t.data[i][j]);
            }
        }
        return tensor;
    }

    public Tensor subi(Tensor t) {
        assertSameLength(t);
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

    private void assertSameLength(Tensor a) {
        if (!Arrays.equals(dimShape, a.shape())) {
            throw new SizeException(String.format("Tensor must have same length (is: {%d,%d,%d,%d} and + {%d,%d,%d,%d})",
                                                    dimShape[0], dimShape[1], dimShape[2], dimShape[3],
                                                    a.dimShape[0], a.dimShape[1], a.dimShape[2], a.dimShape[3]));
        }
    }

    private void assertMatchSize(double[] data, int[] shape) {
        int length = 1;
        for (int i = 0 ; i < shape.length; i++)
            length *= shape[i];

        if (data != null && data.length != length) {
            throw new IllegalArgumentException(
                    "Passed data must match matrix dimensions.");
        }
    }

    public String toString() {
        StringBuilder builder = new StringBuilder();
        for (int i = 0 ; i < dimShape[0]; i++) {
            builder.append(String.format("%d th kernels", i)).append("\n");
            for (int j = 0; j < dimShape[1]; j++) {
                builder.append(String.format("%d th channels", j)).append("\n");
                builder.append(data[i][j].toString()).append("\n");
            }
        }
        return builder.toString();
    }
}
