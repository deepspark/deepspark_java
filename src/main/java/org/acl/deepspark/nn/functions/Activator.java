package org.acl.deepspark.nn.functions;

import org.acl.deepspark.data.Tensor;
import org.jblas.DoubleMatrix;

import java.io.Serializable;

public abstract class Activator implements Serializable {
	public abstract DoubleMatrix output(DoubleMatrix input);
	public abstract DoubleMatrix derivative(DoubleMatrix activated);

	public abstract Tensor output(Tensor input);
	public abstract Tensor derivative(Tensor activated);
}
