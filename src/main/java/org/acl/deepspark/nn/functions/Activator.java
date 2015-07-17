package org.acl.deepspark.nn.functions;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface Activator {
	public INDArray output(INDArray input);
	public INDArray derivative(INDArray input);
}
