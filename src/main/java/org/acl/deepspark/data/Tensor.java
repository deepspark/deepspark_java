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

    private Tensor(int... newDim) {
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
                int startPos = i*dimShape[1]*subArr.length + j*subArr.length;
                System.arraycopy(newData, startPos, subArr, 0, subArr.length);
                data[i][j] = new DoubleMatrix(dimShape[2], dimShape[3], subArr);
            }
        }
    }

    public DoubleMatrix[][] data() {
        return data;
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

    public static Tensor zeros(int... shape) {
        return new Tensor(init.ZEROS, shape);
    }

    public static Tensor ones(int... shape) {
        return new Tensor(init.ONES, shape);
    }

    public static Tensor rand(int... shape) {
        return new Tensor(init.UNIFORM, shape);
    }

    public static Tensor randn(int... shape) {
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

    public Tensor mul(Tensor t) {
        assertSameLength(t);

        Tensor tensor = new Tensor(dimShape);
        for (int i = 0 ; i < dimShape[0]; i++) {
            for (int j = 0; j < dimShape[1]; j++) {
                tensor.data[i][j] = data[i][j].mul(t.data[i][j]);
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

    public Tensor muli(Tensor t) {
        assertSameLength(t);

        for (int i = 0 ; i < dimShape[0]; i++) {
            for (int j = 0; j < dimShape[1]; j++) {
                data[i][j].muli(t.data[i][j]);
            }
        }
        return this;
    }

    public Tensor mmul(Tensor t) {
        assertMultipliesWith(t);
        Tensor tensor = new Tensor(dimShape[0], dimShape[1], dimShape[2], t.dimShape[3]);
        for (int i = 0 ; i < tensor.dimShape[0]; i++) {
            for (int j = 0; j < tensor.dimShape[1]; j++) {
                tensor.data[i][j] = data[i][j].mmul(t.data[i][j]);
            }
        }
        return tensor;
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

    public Tensor div(Tensor t) {
        assertSameLength(t);

        Tensor tensor = new Tensor(dimShape);
        for (int i = 0 ; i < dimShape[0]; i++) {
            for (int j = 0; j < dimShape[1]; j++) {
                tensor.data[i][j] = data[i][j].div(t.data[i][j]);
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

    public Tensor divi(Tensor t) {
        assertSameLength(t);
        for (int i = 0 ; i < dimShape[0]; i++) {
            for (int j = 0; j < dimShape[1]; j++) {
                data[i][j].divi(t.data[i][j]);
            }
        }
        return this;
    }

    public double sum() {
        double sum = 0;
        for (int i = 0 ; i < dimShape[0]; i++) {
            for (int j = 0; j < dimShape[1]; j++) {
                sum += data[i][j].sum();
            }
        }
        return sum;
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

    public Tensor transpose() {
        Tensor t = new Tensor(dimShape[0], dimShape[1], dimShape[3], dimShape[2]);
        for (int i = 0 ; i < dimShape[0]; i++) {
            for (int j = 0; j < dimShape[1]; j++) {
                t.data[i][j] = data[i][j].transpose();
            }
        }
        return t;
    }

    public double[] toArray() {
        double[] arr = new double[length()];
        int channels = dimShape[1];
        int matSize = dimShape[2]*dimShape[3];       // row x col

        for (int i = 0 ; i < dimShape[0]; i++) {
            for (int j = 0; j < dimShape[1]; j++) {
                int startPos = i*channels*matSize + j*matSize;
                System.arraycopy(data[i][j].data, 0, arr, startPos, matSize);
            }
        }
        return arr;
    }

    public Tensor reshape(int... shape) {
        return Tensor.create(toArray(), shape);
    }

    private void assertSameLength(Tensor a) {
        if (!Arrays.equals(dimShape, a.shape())) {
            throw new SizeException(String.format("Tensors must have same length (is: {%d,%d,%d,%d} and {%d,%d,%d,%d})",
                                                    dimShape[0], dimShape[1], dimShape[2], dimShape[3],
                                                    a.dimShape[0], a.dimShape[1], a.dimShape[2], a.dimShape[3]));
        }
    }

    private void assertMatchSize(double[] data, int[] shape) {
        int length = 1;
        for (int i = 0 ; i < shape.length; i++)
            length *= shape[i];

        if (data != null && data.length != length) {
            throw new SizeException(
                    "Passed data must match shape dimensions.");
        }
    }

    private void assertMultipliesWith(Tensor t) {
        if (t.dimShape[0] != dimShape[0] || t.dimShape[1] != dimShape[1]) {
            throw new SizeException(String.format("Tensors must have same kernel and channel size (" +
                                    "is {%d,%d} and {%d,%d}", dimShape[0], dimShape[1], t.dimShape[0], t.dimShape[1]));
        } else {
            if (dimShape[3] != t.dimShape[2])
                throw new SizeException("Number of columns of left matrix must be equal to number of rows of right matrix.");
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
