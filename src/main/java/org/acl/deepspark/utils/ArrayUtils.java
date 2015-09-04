package org.acl.deepspark.utils;

import org.acl.deepspark.data.Tensor;
import org.jblas.DoubleMatrix;
import org.jblas.SimpleBlas;
import org.jblas.ranges.RangeUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.NDArrayFactory;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndexAll;
import org.nd4j.linalg.indexing.NDArrayIndexEmpty;
import org.nd4j.linalg.util.NDArrayUtil;

public class ArrayUtils {
	public static final int FULL_CONV = 0;
	public static final int SAME_CONV = 1;
	public static final int VALID_CONV = 2;
	
    public static INDArray makeColumnVector(INDArray data) {
		INDArray d = data.dup();
		return d.reshape(data.length(), 1);
	}

	public static Tensor makeRowVector(Tensor data) {
		return data.reshape(data.length());
	}
    
    public static INDArray rot90(INDArray toRotate) {
        if (!toRotate.isMatrix())
            throw new IllegalArgumentException("Only rotating matrices");

        INDArray start = toRotate.transpose();
        for (int i = 0; i < start.rows(); i++)
            start.putRow(i, reverse(start.getRow(i)));
        return start;
    }
    
    public static INDArray reverse(INDArray reverse) {
        INDArray rev = reverse.linearView();
        INDArray ret = Nd4j.create(rev.shape());
        int count = 0;
        for (int i = rev.length() - 1; i >= 0; i--) {
            ret.putScalar(count++, rev.getFloat(i));
        }
        return ret.reshape(reverse.shape());
    }
    
    public static int argmax(INDArray arr) {
    	double[] data = arr.data().asDouble();
    	double maxVal = Double.NEGATIVE_INFINITY;
    	int maxIdx = -1;
    	
    	for(int i = 0; i < data.length; i++) {
    		if(data[i] > maxVal) {
    			maxVal = data[i];
    			maxIdx = i;
    		}
    	}
    	return maxIdx;
    }

    public static DoubleMatrix convolution(DoubleMatrix data, DoubleMatrix filter, int option) {
		DoubleMatrix result;
		DoubleMatrix input;

		int nCols, nRows;
		switch(option) {
			case FULL_CONV:
				nRows = data.rows + filter.rows -1;
				nCols = data.columns + filter.columns -1;
				input = DoubleMatrix.zeros(nRows+ filter.rows - 1, nCols + filter.columns - 1);
				input.put(RangeUtils.interval(filter.rows - 1, filter.rows + data.rows - 1),
						  RangeUtils.interval(filter.columns - 1, filter.columns + data.columns - 1), data);

				break;
			case SAME_CONV:
				nRows = data.getRows();
				nCols = data.getColumns();
				input = DoubleMatrix.zeros(nRows+ filter.rows - 1 , nCols + filter.columns - 1);
				input.put(RangeUtils.interval(filter.rows / 2, filter.rows / 2 + data.rows),
						RangeUtils.interval(filter.columns / 2, filter.columns /2  + data.columns), data);
				break;
			case VALID_CONV:
				nRows = data.rows - filter.rows + 1;
				nCols = data.columns - filter.columns + 1;
				input = data;
				break;
			default:
				return null;
		}
		result = DoubleMatrix.zeros(nRows, nCols);
		for(int r = 0; r < nRows ; r++) {
			for(int c = 0 ; c < nCols ; c++) {
				result.put(r,c,
						SimpleBlas.dot(input.get(RangeUtils.interval(r, r + filter.getRows()),
								RangeUtils.interval(c, c + filter.getColumns())), filter));
			}
		}
		return result;
	}

	public static DoubleMatrix flip(DoubleMatrix d) {
		DoubleMatrix output = d.dup();
		for(int k = 0; k < output.getRows() / 2 ; k++)
			output.swapRows(k, output.getRows() - 1 -k);
		for(int k = 0; k < output.getColumns() / 2 ; k++)
			output.swapColumns(k, output.getColumns() - 1 -k);
		return output;
	}
}
