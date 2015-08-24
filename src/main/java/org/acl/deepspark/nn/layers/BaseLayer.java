package org.acl.deepspark.nn.layers;

import org.acl.deepspark.nn.conf.LayerConf;
import org.acl.deepspark.nn.functions.Activator;
import org.acl.deepspark.nn.functions.ActivatorFactory;
import org.acl.deepspark.nn.functions.ActivatorType;

import java.io.Serializable;


public abstract class BaseLayer implements Layer,Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 2727430537685176806L;
	protected int[] inputShape;
	protected Activator activator;
	
	public BaseLayer(int[] shapes, LayerConf conf) {
		inputShape = new int[shapes.length];
		inputShape = shapes.clone();
		activator = ActivatorFactory.getActivator((ActivatorType) conf.get("activator"));
	}
	
	public int[] getInputShape() {
		return inputShape;
	}
}
