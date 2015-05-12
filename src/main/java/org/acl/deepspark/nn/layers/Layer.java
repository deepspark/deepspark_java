package org.acl.deepspark.nn.layers;

import org.jblas.DoubleMatrix;

public interface Layer {
	public DoubleMatrix activate();
}
