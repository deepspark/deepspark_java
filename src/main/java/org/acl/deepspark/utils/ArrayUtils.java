package org.acl.deepspark.utils;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.util.NDArrayUtil;

public class ArrayUtils {
	public static final int FULL_CONV = 0;
	public static final int SAME_CONV = 1;
	public static final int VALID_CONV = 2;
	
    public static INDArray makeColumnVector(INDArray data) {
		INDArray d = data.dup();
		return d.reshape(data.length(), 1);
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
    
    public static INDArray convolution(INDArray data, INDArray filter, int option) {
		INDArray result;
		INDArray input;
		int nCols, nRows;
		switch(option) {
			case FULL_CONV:
				nRows = data.rows() + filter.rows() -1;
				nCols = data.columns() + filter.columns() -1;
				input = Nd4j.zeros(nRows+ filter.rows() -1, nCols + filter.columns() - 1);
				for (int i = 0; i < data.rows(); i++) {
					for (int j = 0; j < data.columns(); j++) {
						input.put(filter.rows() - 1 + i, filter.columns() -1 + j, data.getDouble(i, j));
					}
				}
				break;
		case SAME_CONV:
			nRows = data.rows();
			nCols = data.columns();
			input = Nd4j.zeros(nRows+ filter.rows() - 1 , nCols + filter.columns() - 1);
			input.put(new NDArrayIndex[] {NDArrayIndex.interval(filter.rows() / 2, filter.rows() / 2 + data.rows()), 
					NDArrayIndex.interval(filter.columns() / 2, filter.columns() /2  + data.columns())},
					data);
			break;
			case VALID_CONV:
				nRows = data.rows() - filter.rows() + 1;
				nCols = data.columns() - filter.columns() + 1;
				input = data;
				break;
			default:
				return null;
		}

		result = Nd4j.zeros(nRows, nCols);
		for(int r = 0; r < nRows ; r++) {
			for(int c = 0 ; c < nCols ; c++) {
				INDArray d = input.get(NDArrayIndex.interval(r, r + filter.rows()),
						NDArrayIndex.interval(c, c + filter.columns()));
				result.put(r,c, Nd4j.sum(d.mul(filter)));
			}
		}
		
		return result;
	}
}
