package org.acl.deepspark.nn.functions;

import java.io.Serializable;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;
import org.jblas.SimpleBlas;

public class Activator implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 3874185953299398816L;
	public static final int SIGMOID = 0;
	public static final int TANH = 1;
	public static final int RELU = 2;
	public static final int SOFTMAX =3;
	
	public static double sigmoid(double x) {
		double temp = Math.exp(-x);
		return 1 / (1 + temp);
	}
	
	public static DoubleMatrix sigmoid(DoubleMatrix matrix) {
		if(matrix != null) {
			int rows = matrix.rows;
			int cols = matrix.columns;
			double activation;
			
			for(int m = 0; m < rows; m++) {
				for(int n = 0; n < cols; n++) {
					activation = sigmoid(matrix.get(m,n));
					matrix.put(m, n, activation);
				}
			}
		}
		return matrix;
	}
	
	public static DoubleMatrix[] sigmoid(DoubleMatrix[] matrices) {
		int size = matrices.length;
		for(int i = 0; i < size; i++) {
			matrices[i] = sigmoid(matrices[i]);
		}
		return matrices;
	}
	
	public static double tanh(double x) {
		return Math.tanh(x);
	}
	
	public static DoubleMatrix tanh(DoubleMatrix matrix) {
		if(matrix != null) {
			int rows = matrix.rows;
			int cols = matrix.columns;
			double activation;
			
			for(int m = 0; m < rows; m++) {
				for(int n = 0; n < cols; n++) {
					activation = tanh(matrix.get(m,n));
					matrix.put(m, n, activation);
				}
			}
		}
		return matrix;
	}
	
	
	public static DoubleMatrix[] tanh(DoubleMatrix[] matrices) {
		int size = matrices.length;
		for(int i = 0; i < size; i++) {
			matrices[i] = tanh(matrices[i]);
		}
		return matrices;
	}
	
	public static double relu(double x) {
		return Math.max(0, x);
	}
	
	public static DoubleMatrix relu(DoubleMatrix matrix) {
		if(matrix != null) {
			int rows = matrix.rows;
			int cols = matrix.columns;
			double activation;
			
			for(int m = 0; m < rows; m++) {
				for(int n = 0; n < cols; n++) {
					activation = relu(matrix.get(m,n));
					matrix.put(m, n, activation);
				}
			}
		}
		return matrix;
	}
	
	
	public static DoubleMatrix[] relu(DoubleMatrix[] matrices) {
		int size = matrices.length;
		for(int i = 0; i < size; i++) {
			matrices[i] = relu(matrices[i]);
		}
		return matrices;
	}

	public static DoubleMatrix softmax(DoubleMatrix matrix) {
		if(matrix != null) {
			double max = matrix.max();
			
			matrix.subi(max);
			MatrixFunctions.expi(matrix);
			double sum = matrix.sum();
			matrix.divi(sum);
		}
		return matrix;
	}
}
