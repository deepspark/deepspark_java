package org.acl.deepspark.nn.layers;

import org.jblas.DoubleMatrix;

public abstract class BaseLayer {
	public abstract double[] getOutput();
	public abstract void update(DoubleMatrix[] weights); 
}
