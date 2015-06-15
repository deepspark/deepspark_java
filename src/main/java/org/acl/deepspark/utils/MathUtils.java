package org.acl.deepspark.utils;

import java.io.Serializable;

import org.jblas.DoubleMatrix;
import org.jblas.SimpleBlas;
import org.jblas.ranges.RangeUtils;

public class MathUtils  implements Serializable{
	/**
	 * 
	 */
	private static final long serialVersionUID = -3873474318160905207L;
	public static final int FULL_CONV = 0;
	public static final int SAME_CONV = 1;
	public static final int VALID_CONV = 2;
	
	
	
	public static DoubleMatrix convolution(DoubleMatrix data, DoubleMatrix filter, int option) {
		DoubleMatrix result;
		DoubleMatrix input;
		int nCols, nRows;
		switch(option) {
		case FULL_CONV:
			nRows = data.getRows() + filter.getRows() - 1;
			nCols = data.getColumns() + filter.getColumns() - 1;
			input = DoubleMatrix.zeros(nRows+ filter.getRows() -1, nCols + filter.getColumns() - 1);
			input.put(RangeUtils.interval(filter.getRows() - 1, filter.getRows() + data.getRows() - 1), 
					RangeUtils.interval(filter.getColumns() - 1, filter.getColumns() + data.getColumns() - 1),
					data);
			break;
		case SAME_CONV:
			nRows = data.getRows();
			nCols = data.getColumns();
			input = DoubleMatrix.zeros(nRows+ filter.getRows() - 1 , nCols + filter.getColumns() - 1);
			input.put(RangeUtils.interval(filter.getRows() / 2, filter.getRows() / 2 + data.getRows()), 
					RangeUtils.interval(filter.getColumns() / 2, filter.getColumns() /2  + data.getColumns()),
					data);
			break;
		case VALID_CONV:
			nRows = data.getRows() - filter.getRows() + 1;
			nCols = data.getColumns() - filter.getColumns() + 1;
			input = data;
			break;
		default:
			return null;
		}
		
		result = new DoubleMatrix(nRows, nCols);
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
		for(int k = 0; k < d.getRows() / 2 ; k++)
			d.swapRows(k, d.getRows() - 1 -k);
		for(int k = 0; k < d.getColumns() / 2 ; k++)
			d.swapRows(k, d.getRows() - 1 -k);

		return d;
	}
}
