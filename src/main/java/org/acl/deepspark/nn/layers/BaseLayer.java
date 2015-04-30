package org.acl.deepspark.nn.layers;

public abstract class BaseLayer {
	public abstract double[] getOutput();
	public abstract void update(double[] weights); 
}
