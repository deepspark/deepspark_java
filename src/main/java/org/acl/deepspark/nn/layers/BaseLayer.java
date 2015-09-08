package org.acl.deepspark.nn.layers;

import java.io.Serializable;


public abstract class BaseLayer implements Layer,Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 2727430537685176806L;
	private int[] inputShape;

	
	public BaseLayer(int[] shapes) {
		inputShape = shapes.clone();
	}
	
	public int[] getInputShape() {
		return inputShape;
	}
}
