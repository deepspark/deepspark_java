package org.acl.deepspark.nn.layers;

import java.io.Serializable;


public abstract class BaseLayer implements Layer,Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 2727430537685176806L;
	private int[] inputShape;
	
	public BaseLayer(int[] shapes) {
		inputShape = new int[shapes.length];
		System.arraycopy(shapes, 0, inputShape, 0, shapes.length);
	}
	
	public int[] getInputShape() {
		return inputShape;
	}
}
