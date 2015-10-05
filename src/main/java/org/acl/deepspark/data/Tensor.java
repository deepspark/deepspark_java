package org.acl.deepspark.data;

import org.acl.deepspark.utils.GPUUtils;
import org.jblas.FloatMatrix;
import org.jblas.exceptions.SizeException;

import java.io.Serializable;

/**
 * Created by Jaehong on 2015-09-02.
 */
public class Tensor implements Serializable {

    protected int[] dimShape;      // dimShape = {kernels, channels, rows, cols}
    protected int length;          // length = kernels * channels
    protected FloatMatrix[] data;  // data = DoubleMatrix[kernels * channels]

    public enum init {
        ZEROS, ONES, UNIFORM, GAUSSIAN, XAVIER
    }

    protected Tensor() {
        dimShape = new int[] {1, 1, 1, 1};
    }

    protected Tensor(int... newDim) {
        this();
        if (newDim != null) {
            if (newDim.length > 4)
                throw new IllegalStateException(String.format("Only support (n <= 4) dimensional tensor, current: %d", newDim.length));
        /* dimShape = {kernels, channels, rows, cols} */
            System.arraycopy(newDim, 0, dimShape, 4-newDim.length, newDim.length);
            length = dimShape[0]*dimShape[1];
            data = new FloatMatrix[length];
        }
    }

    protected Tensor(Tensor.init init, int[] newDim) {
        this(newDim);
        for (int i = 0; i < length; i++) {
            switch (init) {
                case ZEROS:
                    data[i] = FloatMatrix.zeros(dimShape[2], dimShape[3]);
                    break;

                case ONES:
                    data[i] = FloatMatrix.ones(dimShape[2], dimShape[3]);
                    break;

                case UNIFORM:
                    data[i] = FloatMatrix.rand(dimShape[2], dimShape[3]);
                    break;

                case GAUSSIAN:
                    data[i] = FloatMatrix.randn(dimShape[2], dimShape[3]);
                    break;
            }
        }
    }

    protected Tensor(float[] newData, int[] newDim) {
        this(newDim);
        assertMatchSize(newData, newDim);

        int matSize = dimShape[2]*dimShape[3];
        for (int i = 0 ; i < length; i++) {
            float[] subArr = new float[matSize];
            System.arraycopy(newData, i*matSize, subArr, 0, subArr.length);
            data[i] = new FloatMatrix(dimShape[2], dimShape[3], subArr);
        }
    }

    protected Tensor(FloatMatrix[] newData, int[] newDim) {
        this(newDim);
        if (length != newData.length)
            throw new SizeException(String.format("Input data length(%d) must match with Tensor size(%d)", newData.length, length));
        for (FloatMatrix mat : newData)
            mat.reshape(dimShape[2], dimShape[3]);
        data = newData;
    }

    public FloatMatrix[] data() {
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

    public FloatMatrix slice(int kernelIdx) {
        return slice(index(kernelIdx, 0));
    }

    public FloatMatrix slice(int kernelIdx, int channelIdx) {
        return data[index(kernelIdx, channelIdx)];
    }

    public static Tensor create(float[] newData, int[] newDim) {
        return new Tensor(newData, newDim);
    }

    public static Tensor create(FloatMatrix[] newData, int[] newDim) {
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

    public Tensor add(float d) {
        Tensor tensor = new Tensor(dimShape);
        for (int i = 0 ; i < length; i++)
            tensor.data[i] = data[i].add(d);
        return tensor;
    }

    public Tensor add(FloatMatrix matrix) {
        Tensor tensor = new Tensor(dimShape);
        for (int i = 0 ; i < length; i++)
            tensor.data[i] = data[i].add(matrix);
        return tensor;
    }

    public Tensor add(Tensor t) {
        assertSameLength(t);

        Tensor tensor = new Tensor(dimShape);
        for (int i = 0; i < length; i++)
            tensor.data[i] = data[i].add(t.data[i]);
        return tensor;
    }

    public Tensor addRowVector(FloatMatrix mat) {
        Tensor ret = new Tensor(dimShape);
        for (int i = 0 ; i < data.length; i++)
            ret.data[i] = data[i].addRowVector(mat);
        return ret;
    }

    public Tensor addRowTensor(Tensor t) {
        assertSameLength(t);

        Tensor ret = new Tensor(dimShape);
        for (int i = 0 ; i < data.length; i++)
            ret.data[i] = data[i].addRowVector(t.data[i]);
        return ret;
    }

    public Tensor addColumnVector(FloatMatrix mat) {
        Tensor ret = new Tensor(dimShape);
        for (int i = 0 ; i < data.length; i++)
            ret.data[i] = data[i].addColumnVector(mat);
        return ret;
    }

    public Tensor addColumnTensor(Tensor t) {
        assertSameLength(t);

        Tensor ret = new Tensor(dimShape);
        for (int i = 0 ; i < data.length; i++)
            ret.data[i] = data[i].addColumnVector(t.data[i]);
        return ret;
    }

    public Tensor addi(float d) {
        for (int i = 0 ; i < length; i++)
            data[i].addi(d);
        return this;
    }

    public Tensor addi(FloatMatrix mat) {
        for (int i = 0 ; i < length; i++)
            data[i].addi(mat);
        return this;
    }

    public Tensor addi(Tensor t) {
        assertSameLength(t);
        for (int i = 0 ; i < length; i++)
            data[i].addi(t.data[i]);
        return this;
    }

    public Tensor addiRowVector(FloatMatrix mat) {
        for (int i = 0 ; i < length; i++)
            data[i].addiRowVector(mat);
        return this;
    }

    public Tensor addiRowTensor(Tensor t) {
        assertSameLength(t);
        for (int i = 0 ; i < data.length; i++)
            data[i].addiRowVector(t.data[i]);
        return this;
    }

    public Tensor addiColumnVector(FloatMatrix mat) {
        for (int i = 0 ; i < length; i++)
            data[i].addiColumnVector(mat);
        return this;
    }

    public Tensor addiColumnTensor(Tensor t) {
        assertSameLength(t);
        for (int i = 0 ; i < data.length; i++)
            data[i].addiColumnVector(t.data[i]);
        return this;
    }

    public Tensor sub(float d) {
        Tensor tensor = new Tensor(dimShape);
        for (int i = 0 ; i < length; i++)
            tensor.data[i] = data[i].sub(d);
        return tensor;
    }

    public Tensor sub(FloatMatrix matrix) {
        Tensor tensor = new Tensor(dimShape);
        for (int i = 0 ; i < length; i++)
            tensor.data[i] = data[i].sub(matrix);
        return tensor;
    }

    public Tensor sub(Tensor t) {
        assertSameLength(t);

        Tensor tensor = new Tensor(dimShape);
        for (int i = 0; i < length; i++)
            tensor.data[i] = data[i].sub(t.data[i]);
        return tensor;
    }

    public Tensor subRowTensor(Tensor t) {
        assertSameLength(t);

        Tensor ret = new Tensor(dimShape);
        for (int i = 0 ; i < data.length; i++)
            ret.data[i] = data[i].subRowVector(t.data[i]);
        return ret;
    }

    public Tensor subColumnTensor(Tensor t) {
        assertSameLength(t);

        Tensor ret = new Tensor(dimShape);
        for (int i = 0 ; i < data.length; i++)
            ret.data[i] = data[i].subColumnVector(t.data[i]);
        return ret;
    }

    public Tensor subi(float d) {
        for (int i = 0 ; i < length; i++)
            data[i].subi(d);
        return this;
    }

    public Tensor subi(FloatMatrix mat) {
        for (int i = 0 ; i < length; i++)
            data[i].subi(mat);
        return this;
    }

    public Tensor subi(Tensor t) {
        assertSameLength(t);
        for (int i = 0 ; i < length; i++)
            data[i].subi(t.data[i]);
        return this;
    }

    public Tensor subiRowTensor(Tensor t) {
        assertSameLength(t);
        for (int i = 0 ; i < data.length; i++)
            data[i].subiRowVector(t.data[i]);
        return this;
    }

    public Tensor subiColumnTensor(Tensor t) {
        assertSameLength(t);
        for (int i = 0 ; i < data.length; i++)
            data[i].subiColumnVector(t.data[i]);
        return this;
    }

    public Tensor mul(float d) {
        Tensor tensor = new Tensor(dimShape);
        for (int i = 0 ; i < length; i++)
            tensor.data[i] = data[i].mul(d);
        return tensor;
    }

    public Tensor mul(FloatMatrix matrix) {
        Tensor tensor = new Tensor(dimShape);
        for (int i = 0 ; i < length; i++)
            tensor.data[i] = data[i].mul(matrix);
        return tensor;
    }

    public Tensor mul(Tensor t) {
        assertSameLength(t);

        Tensor tensor = new Tensor(dimShape);
        for (int i = 0; i < length; i++)
            tensor.data[i] = data[i].mul(t.data[i]);
        return tensor;
    }

    public Tensor mulRowTensor(Tensor t) {
        assertSameLength(t);

        Tensor ret = new Tensor(dimShape);
        for (int i = 0 ; i < length; i++)
            ret.data[i] = data[i].mulRowVector(t.data[i]);
        return ret;
    }

    public Tensor mulColumnTensor(Tensor t) {
        assertSameLength(t);

        Tensor ret = new Tensor(dimShape);
        for (int i = 0 ; i < length; i++)
            ret.data[i] = data[i].mulColumnVector(t.data[i]);
        return ret;
    }

    public Tensor muli(float d) {
        for (int i = 0 ; i < length; i++)
            data[i].muli(d);
        return this;
    }

    public Tensor muli(FloatMatrix mat) {
        for (int i = 0 ; i < length; i++)
            data[i].muli(mat);
        return this;
    }

    public Tensor muli(Tensor t) {
        assertSameLength(t);
        for (int i = 0 ; i < length; i++)
            data[i].muli(t.data[i]);
        return this;
    }

    public Tensor muliRowTensor(Tensor t) {
        assertSameLength(t);
        for (int i = 0 ; i < data.length; i++)
            data[i].muliRowVector(t.data[i]);
        return this;
    }

    public Tensor muliColumnTensor(Tensor t) {
        assertSameLength(t);
        for (int i = 0 ; i < data.length; i++)
            data[i].muliColumnVector(t.data[i]);
        return this;
    }

    public Tensor mmul(FloatMatrix matrix) {
        Tensor tensor = new Tensor(dimShape[0], dimShape[1], dimShape[2], matrix.columns);
        for (int i = 0 ; i < length; i++)
            tensor.data[i] = data[i].mmul(matrix);
        return tensor;
    }

    public Tensor mmul(Tensor t) {
        assertMultipliesWith(t);
        Tensor tensor = new Tensor(dimShape[0], dimShape[1], dimShape[2], t.dimShape[3]);
        for (int i = 0 ; i < length; i++)
            tensor.data[i] = data[i].mmul(t.data[i]);
        return tensor;
    }

//    public Tensor mmul(Tensor other) {
//        assertMultipliesWith(other);
//
//        Tensor result = new Tensor(dimShape[0], dimShape[1], dimShape[2], other.dimShape[3]);
//
//        for (int i = 0; i < data.length; i++) {
//            result.data[i] = new FloatMatrix(data[i].rows, other.data[i].columns);
//            GPUUtils.sgemmJCublas('n', 'n', 1, data[i], other.data[i], 0, result.data[i]);
//        }
//
//        return result;
//    }

    public Tensor div(float d) {
        Tensor tensor = new Tensor(dimShape);
        for (int i = 0 ; i < length; i++)
            tensor.data[i] = data[i].div(d);
        return tensor;
    }

    public Tensor div(FloatMatrix matrix) {
        Tensor tensor = new Tensor(dimShape);
        for (int i = 0 ; i < length; i++)
            tensor.data[i] = data[i].div(matrix);
        return tensor;
    }

    public Tensor div(Tensor t) {
        assertSameLength(t);

        Tensor tensor = new Tensor(dimShape);
        for (int i = 0; i < length; i++)
            tensor.data[i] = data[i].div(t.data[i]);
        return tensor;
    }

    public Tensor divRowTensor(Tensor t) {
        assertSameLength(t);

        Tensor ret = new Tensor(dimShape);
        for (int i = 0 ; i < length; i++)
            ret.data[i] = data[i].divRowVector(t.data[i]);
        return ret;
    }

    public Tensor divColumnTensor(Tensor t) {
        assertSameLength(t);
        Tensor ret = new Tensor(dimShape);
        for (int i = 0; i < length; i++)
            ret.data[i] = data[i].divColumnVector(t.data[i]);
        return ret;
    }

    public Tensor divi(float d) {
        for (int i = 0 ; i < length; i++)
            data[i].divi(d);
        return this;
    }

    public Tensor divi(FloatMatrix mat) {
        for (int i = 0 ; i < length; i++)
            data[i].divi(mat);
        return this;
    }

    public Tensor divi(Tensor t) {
        assertSameLength(t);
        for (int i = 0 ; i < length; i++)
            data[i].divi(t.data[i]);
        return this;
    }

    public Tensor diviRowTensor(Tensor t) {
        assertSameLength(t);
        for (int i = 0 ; i < length; i++)
            data[i].diviRowVector(t.data[i]);
        return this;
    }

    public Tensor diviColumnTensor(Tensor t) {
        assertSameLength(t);
        for (int i = 0 ; i < length; i++)
            data[i].diviColumnVector(t.data[i]);
        return this;
    }

    public Tensor transpose() {
        Tensor t = new Tensor(dimShape[0], dimShape[1], dimShape[3], dimShape[2]);
        for (int i = 0 ; i < length; i++)
            t.data[i] = data[i].transpose();
        return t;
    }
/*
    public static Tensor flatToColumnTensor(Tensor t) {

    }
*/

    public float sum(int kernelIdx, int channelIdx) {
        return data[index(kernelIdx, channelIdx)].sum();
    }

    public float sum() {
        float sum = 0;
        for (int i = 0 ; i < length; i++)
            sum += data[i].sum();
        return sum;
    }

    public Tensor mean() {
        Tensor ret = new Tensor(dimShape);
        for (int i = 0 ; i < length; i++)
            ret.data[i] = FloatMatrix.ones(dimShape[2], dimShape[3]).muli(data[i].mean());
        return ret;
    }

    public Tensor rowSums() {
        Tensor ret = new Tensor(dimShape[0], dimShape[1], dimShape[2], 1);
        for (int i = 0 ; i < length; i++)
            ret.data[i] = data[i].rowSums();
        return ret;
    }

    public Tensor columnSums() {
        Tensor ret = new Tensor(dimShape[0], dimShape[1], 1, dimShape[3]);
        for (int i = 0 ; i < length; i++)
            ret.data[i] = data[i].columnSums();
        return ret;
    }

    private int index(int kernelIdx, int channelIdx) {
        return kernelIdx*dimShape[1] + channelIdx;
    }

    public Tensor dup() {
        Tensor tensor = new Tensor(dimShape.clone());
        System.arraycopy(data, 0, tensor.data, 0, length);
        return tensor;
    }

    public static Tensor merge(Tensor... tensors) {
        // merged Tensors must have same shapes
        for (Tensor t : tensors)
            tensors[0].assertSameShape(t);
        Tensor ret = new Tensor(tensors[0].dimShape[0]*tensors.length, tensors[0].dimShape[1],
                tensors[0].dimShape[2], tensors[0].dimShape[3]);

        int dataSize = tensors[0].length;
        for (int i = 0 ; i < tensors.length; i++)
            System.arraycopy(tensors[i].data, 0, ret.data, i*dataSize, dataSize);
        return ret;
    }

    public static Tensor mergei(Tensor... tensors) {
        for (Tensor t : tensors)
            tensors[0].assertSameLength(t);
        Tensor ret = new Tensor(tensors[0].dimShape[0]*tensors.length, tensors[0].dimShape[1],
                tensors[0].dimShape[2], tensors[0].dimShape[3]);

        int idx = 0;
        for (int i = 0 ; i < tensors.length; i++) {
            for (int j = 0 ; j < tensors[0].length; j++)
                ret.data[idx++] = tensors[i].data[j];
        }
        return ret;
    }

    public FloatMatrix addAll() {
        FloatMatrix ret = FloatMatrix.zeros(dimShape[2], dimShape[3]);
        for (FloatMatrix mat : data)
            ret.addi(mat);
        return ret;
    }


    public float[] toArray() {
        float[] arr = new float[length()];
        int matSize = dimShape[2]*dimShape[3];       // row x col

        for (int i = 0 ; i < length; i++)
            System.arraycopy(data[i].data, 0, arr, i*matSize, matSize);
        return arr;
    }

    public Tensor reshape(int... shape) {
        return Tensor.create(toArray(), shape);
    }

    protected void assertSameLength(Tensor a) {
        if (length != a.length)
            throw new SizeException(String.format("Tensors must have same length (is: {%d} and {%d})", length, a.length));
    }

    protected void assertSameShape(Tensor t) {
        if (dimShape[0] != t.dimShape[0] || dimShape[1] != t.dimShape[1] ||
                dimShape[2] != t.dimShape[2] || dimShape[3] != t.dimShape[3])
            throw new SizeException(String.format("Tensors must have same shape (is: {%d, %d, %d, %d} and {%d, %d, %d, %d}",
                    dimShape[0], dimShape[1], dimShape[2], dimShape[3], t.dimShape[0], t.dimShape[1], t.dimShape[2], t.dimShape[3]));
    }

    protected void assertMatchSize(float[] data, int[] shape) {
        int length = 1;
        for (int i = 0 ; i < shape.length; i++)
            length *= shape[i];

        if (data != null && data.length != length) {
            throw new SizeException(
                    "Passed data must match shape dimensions.");
        }
    }

    protected void assertMultipliesWith(Tensor t) {
        assertSameLength(t);
        if (dimShape[3] != t.dimShape[2])
            throw new SizeException(String.format("Number of columns of left matrix (%d) must be equal to number of rows of right matrix (%d).", dimShape[3], t.dimShape[2]));
    }

    public String toString() {
        StringBuilder builder = new StringBuilder();
        for (int i = 0 ; i < dimShape[0]; i++) {
            builder.append(String.format("%d th kernels", i)).append("\n");
            for (int j = 0; j < dimShape[1]; j++) {
                builder.append(String.format("%d th channels", j)).append("\n");
                builder.append(data[i*dimShape[1] + j].toString()).append("\n");
            }
        }
        return builder.toString();
    }
}
