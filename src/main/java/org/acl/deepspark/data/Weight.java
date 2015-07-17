package org.acl.deepspark.data;

import java.io.Serializable;

import org.nd4j.linalg.api.ndarray.INDArray;

public class Weight implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = -2016361466768395491L;
	
	public INDArray w;
	public INDArray b;	
}
