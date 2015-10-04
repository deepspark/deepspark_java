package org.acl.deepspark.nn.layers;

import java.io.Serializable;


public abstract class BaseLayer implements Layer,Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 2727430537685176806L;
	private int[] dimIn;

	
	public BaseLayer(int[] shapes) {
		dimIn = shapes;
	}
	
	public int[] getDimIn() {
		return dimIn;
	}

}
