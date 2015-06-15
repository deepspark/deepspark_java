package org.acl.deepspark.data;

import java.io.Serializable;

import org.jblas.DoubleMatrix;

public class Sample implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 258491956070013844L;
	public DoubleMatrix[] data;
	public DoubleMatrix label;
}
