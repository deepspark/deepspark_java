package org.acl.deepspark.utils;

import java.util.Arrays;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

public class ArrayUtils {
    public static INDArray padWithZeros(INDArray nd, int[] targetShape) {
        if (Arrays.equals(nd.shape(), targetShape))
            return nd;
        //no padding required
        if (ArrayUtil.prod(nd.shape()) >= ArrayUtil.prod(targetShape))
            return nd;

        INDArray ret = Nd4j.create(targetShape);
        double[] d = ret.data().asDouble();
        System.arraycopy(nd.data().asDouble(), 0, d, 0, nd.data().length());
        ret.data().setData(d);
        return ret;
    }
    
    public static INDArray makeColumnVector(INDArray data) {
		INDArray d = data.dup();
		return d.reshape(data.length(), 1);
	}
}
