package org.acl.deepspark.utils;

import org.acl.deepspark.data.Tensor;
import org.jblas.FloatMatrix;
import org.jblas.SimpleBlas;
import org.jblas.ranges.RangeUtils;

public class ArrayUtils {
	public static final int FULL_CONV = 0;
	public static final int SAME_CONV = 1;
	public static final int VALID_CONV = 2;

	public static Tensor makeRowVector(Tensor in) {
		return in.reshape(in.length());
	}

	public static Tensor padding(Tensor in, int zeroPadding) {
		int length = in.data().length;
		int[] shape = in.shape();
		Tensor ret = Tensor.zeros(shape[0], shape[1], shape[2]+2*zeroPadding, shape[3]+2*zeroPadding);

		for (int i = 0 ; i < length; i++) {
			ret.data()[i].put(RangeUtils.interval(zeroPadding, zeroPadding+shape[2]), RangeUtils.interval(zeroPadding, zeroPadding+shape[3]),
					in.data()[i]);
		}
		return ret;
	}

    public static FloatMatrix convolution(FloatMatrix data, FloatMatrix filter, int option) {
		FloatMatrix result;
		FloatMatrix input;

		int nCols, nRows;
		switch(option) {
			case FULL_CONV:
				nRows = data.rows + filter.rows -1;
				nCols = data.columns + filter.columns -1;
				input = FloatMatrix.zeros(nRows+ filter.rows - 1, nCols + filter.columns - 1);
				input.put(RangeUtils.interval(filter.rows - 1, filter.rows + data.rows - 1),
						  RangeUtils.interval(filter.columns - 1, filter.columns + data.columns - 1), data);

				break;
			case SAME_CONV:
				nRows = data.getRows();
				nCols = data.getColumns();
				input = FloatMatrix.zeros(nRows+ filter.rows - 1 , nCols + filter.columns - 1);
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
		result = FloatMatrix.zeros(nRows, nCols);
		for(int r = 0; r < nRows ; r++) {
			for(int c = 0 ; c < nCols ; c++) {
				result.put(r,c,
						SimpleBlas.dot(input.get(RangeUtils.interval(r, r + filter.getRows()),
								RangeUtils.interval(c, c + filter.getColumns())), filter));
			}
		}
		return result;
	}

	public static FloatMatrix flip(FloatMatrix d) {
		FloatMatrix output = d.dup();
		for(int k = 0; k < output.getRows() / 2 ; k++)
			output.swapRows(k, output.getRows() - 1 -k);
		for(int k = 0; k < output.getColumns() / 2 ; k++)
			output.swapColumns(k, output.getColumns() - 1 -k);
		return output;
	}
}
