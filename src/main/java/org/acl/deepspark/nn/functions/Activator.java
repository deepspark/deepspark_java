package org.acl.deepspark.nn.functions;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;

public abstract class Activator implements Serializable {
	public abstract INDArray output(INDArray input);
	public abstract INDArray derivative(INDArray input, boolean output);
}
