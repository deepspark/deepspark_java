package org.acl.deepspark.utils;

import org.jblas.DoubleMatrix;

public class MathUtils {
	public static final int FULL_CONV = 0;
	public static final int SAME_CONV = 1;
	public static final int VALID_CONV = 2;
	
	
	
	public static DoubleMatrix convolution(DoubleMatrix data, DoubleMatrix filter, int option) {
		DoubleMatrix result;
		DoubleMatrix input;
		int nCols, nRows;
		switch(option) {
		case FULL_CONV:
			nRows = data.getRows() + filter.getRows() + 1;
			nCols = data.getColumns() + filter.getColumns() + 1;
			break;
		case SAME_CONV:
			nRows = data.getRows();
			nCols = data.getColumns();
			input = data;
			break;
		case VALID_CONV:
			nRows = data.getRows();
			nCols = data.getColumns();
			break;
		default:
			return null;
		}
		result = new DoubleMatrix(nRows, nCols);
		return result;
	}
}
