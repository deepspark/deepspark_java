package org.acl.deepspark.nn.layers.cnn;

import org.acl.deepspark.nn.layers.BaseLayer;
import org.jblas.DoubleMatrix;
import org.jblas.ranges.RangeUtils;

public class PoolingLayer extends BaseLayer {
	private DoubleMatrix[] input;
	private int[] poolSize = new int[2];
	
	public PoolingLayer(DoubleMatrix[] input, int poolSize) {
		super();
		this.input = input;
		this.poolSize[0] = this.poolSize[1] = poolSize;
	}
	
	public PoolingLayer(DoubleMatrix input, int poolSize) {
		super();
		this.input = new DoubleMatrix[1];
		this.input[0] = input;
		this.poolSize[0] = this.poolSize[1] = poolSize;
	}
	
	// Apply maxPooling on the featureMap
	// TODO : add min/average pooling
	public DoubleMatrix pooling(DoubleMatrix image) {
		int outputRows = image.rows / poolSize[0];
		int outputCols = image.columns / poolSize[1];
		DoubleMatrix data = new DoubleMatrix(outputRows, outputCols);
		
		for(int m = 0; m < outputRows; m++) {
			for(int n = 0; n < outputCols; n++) {
				data.put(m, n, 
						image.get(RangeUtils.interval(m*poolSize[0], (m+1)*poolSize[0]), 
						  RangeUtils.interval(n*poolSize[1], (n+1)*poolSize[1])).max());
			}
		}
		return data;
	}
	
	public DoubleMatrix[] pooling(DoubleMatrix[] image) {
		int size = image.length;
		DoubleMatrix[] data = new DoubleMatrix[size];
		for(int i = 0; i < size; i++) {
			data[i] = pooling(image[i]);
		}
		return data;
	}
	
	public DoubleMatrix[] pooling() {
		int size = input.length;
		DoubleMatrix[] data = new DoubleMatrix[size];
		for(int i = 0; i < size; i++) {
			data[i] = pooling(input[i]);
		}
		return data;
	}

	@Override
	public DoubleMatrix[] getOutput() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void update(DoubleMatrix[] weights) {
		// TODO Auto-generated method stub
		
	}
}
