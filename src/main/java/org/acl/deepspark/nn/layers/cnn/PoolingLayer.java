package org.acl.deepspark.nn.layers.cnn;

import org.acl.deepspark.nn.layers.BaseLayer;
import org.jblas.DoubleMatrix;
import org.jblas.ranges.RangeUtils;

public class PoolingLayer extends BaseLayer {
	private DoubleMatrix[] input;
	private int inputRows;
	private int inputCols;
	private int outputRows;
	private int outputCols;

	public int[][] maxIndices;
	private int poolSize;
	
	public PoolingLayer(DoubleMatrix[] input, int poolSize) {
		super();
		this.input = input;
		this.inputRows = input[0].rows;
		this.inputCols = input[0].columns;
		this.poolSize = poolSize;
		this.outputRows = inputRows / poolSize;
		this.outputCols = inputCols / poolSize;
		maxIndices = new int [input.length][outputRows * outputCols];
	}
	
	public PoolingLayer(DoubleMatrix input, int poolSize) {
		super();
		this.input = new DoubleMatrix[1];
		this.input[0] = input;
		this.inputRows = input.rows;
		this.inputCols = input.columns;
		this.poolSize = poolSize;
		this.outputRows = inputRows / poolSize;
		this.outputCols = inputCols / poolSize;
		maxIndices = new int [input.length][outputRows * outputCols];
	}
	
	// Apply maxPooling on the featureMap
	// TODO : add min/average pooling
	public DoubleMatrix pooling(DoubleMatrix image) {
		if (image.isEmpty())
			return null;

		DoubleMatrix data = new DoubleMatrix(outputRows, outputCols);
		
		for (int m = 0; m < outputRows; m++) {
			for (int n = 0; n < outputCols; n++) {
				data.put(m, n, 
						image.get(RangeUtils.interval(m*poolSize, (m+1)*poolSize), 
								  RangeUtils.interval(n*poolSize, (n+1)*poolSize)).max());
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
		DoubleMatrix[] data = new DoubleMatrix[input.length];
		double max;
		int maxIdx = 0;
		int idx;
		
		for (int i = 0 ; i < input.length; i++) {
			data[i] = new DoubleMatrix(outputRows, outputCols);
			idx = 0;
			for (int m = 0; m < outputRows; m++) {
				for (int n = 0; n < outputCols; n++) {
					
					max = Double.NEGATIVE_INFINITY;
					for (int r = m*poolSize; r < (m+1)*poolSize; r++) {
						for (int c = n*poolSize; c < (n+1)*poolSize; c++) {
							if (!Double.isNaN(input[i].get(r, c)) && input[i].get(r, c) > max) {
								maxIdx = input[i].index(r, c);
								max = input[i].get(r, c);
							}
						}
					}
					data[i].put(m, n, max);
					maxIndices[i][idx++] = maxIdx;
				}
			}
		}
		return data;
	}

	@Override
	public DoubleMatrix[] getOutput() {
		// TODO Auto-generated method stub
		return activate(pooling());
	}
	
	private int getMaxIndex(int size, int outputRowIdx, int outputColIdx) {
		int dimOut = outputRowIdx * outputColIdx;
		return (size * dimOut) + outputRowIdx * outputCols + outputColIdx; 
	}
	

	@Override
	public DoubleMatrix[] update(DoubleMatrix[] outputDelta) {
		// TODO Auto-generated method stub
		int size = outputDelta.length;
		DoubleMatrix[] inputDelta = new DoubleMatrix[size];
		
		for (int i = 0; i < size; i++) {
			inputDelta[i] = new DoubleMatrix(inputRows, inputCols);
			int idx = 0;
			for (int m = 0; m < outputRows; m++) {
				for (int n = 0; n < outputCols; n++) {
					inputDelta[i].put(maxIndices[i][idx++], outputDelta[i].get(m,n));
				}
			}
		}
		return inputDelta;
	}
}
