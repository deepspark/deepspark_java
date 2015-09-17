package org.acl.deepspark.nn.functions;

import org.acl.deepspark.data.Tensor;
import org.jblas.FloatMatrix;

import java.io.Serializable;

public abstract class Activator implements Serializable {
	public abstract FloatMatrix output(FloatMatrix input);
	public abstract FloatMatrix derivative(FloatMatrix activated);

	public abstract Tensor output(Tensor input);
	public abstract Tensor derivative(Tensor activated);
}
