package org.acl.deepspark.utils;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class ArrayUtils {
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
}
