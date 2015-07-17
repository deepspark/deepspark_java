package org.acl.deepspark.utils;

import org.nd4j.linalg.api.ndarray.INDArray;

public class ArrayUtils {
    public static INDArray makeColumnVector(INDArray data) {
		INDArray d = data.dup();
		return d.reshape(data.length(), 1);
	}
}
