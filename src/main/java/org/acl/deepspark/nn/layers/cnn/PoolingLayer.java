package org.acl.deepspark.nn.layers.cnn;

import org.acl.deepspark.nn.layers.BaseLayer;
import org.jblas.DoubleMatrix;
import org.jblas.ranges.RangeUtils;

public class PoolingLayer extends BaseLayer {
	private int outputRows;
	private int outputCols;

	public int[][] maxIndices;
	private int poolSize;
	
	
	/** Modified **/
	public PoolingLayer(int poolSize) {
		super();
		this.poolSize = poolSize;
	}
	
	public PoolingLayer(DoubleMatrix input, int poolSize) {
		super(input);
		this.poolSize = poolSize;
		this.outputRows = dimRows / poolSize;
		this.outputCols = dimCols / poolSize;
		maxIndices = new int [numChannels][outputRows * outputCols];
	}
	
	public PoolingLayer(DoubleMatrix[] input, int poolSize) {
		super(input);
		this.poolSize = poolSize;
		this.outputRows = dimRows / poolSize;
		this.outputCols = dimCols / poolSize;
		maxIndices = new int [numChannels][outputRows * outputCols];
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
		DoubleMatrix[] data = new DoubleMatrix[numChannels];
		double max;
		int maxIdx = 0;
		int idx;
		
		
		/** Modified **/
		int outputRows = dimRows / poolSize;
		int outputCols = dimCols / poolSize;
		if(maxIndices == null)
			maxIndices = new int [numChannels][outputRows * outputCols];
		
		for (int i = 0 ; i < numChannels; i++) {
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
		return pooling();
	}
	
	private int getMaxIndex(int size, int outputRowIdx, int outputColIdx) {
		int dimOut = outputRowIdx * outputColIdx;
		return (size * dimOut) + outputRowIdx * outputCols + outputColIdx; 
	}
	
	@Override
	public DoubleMatrix[] deriveDelta(DoubleMatrix[] outputDelta) {
		// TODO Auto-generated method stub
		return null;
	}
	

	@Override
	public DoubleMatrix[] update(DoubleMatrix[] outputDelta) {
		// TODO Auto-generated method stub
		DoubleMatrix[] inputDelta = new DoubleMatrix[numChannels];
		
		for (int i = 0; i < numChannels; i++) {
			inputDelta[i] = new DoubleMatrix(dimRows, dimCols);
			int idx = 0;
			for (int m = 0; m < outputRows; m++) {
				for (int n = 0; n < outputCols; n++) {
					inputDelta[i].put(maxIndices[i][idx++], outputDelta[i].get(m,n));
				}
			}
		}
		return inputDelta;
	}

	@Override
	public void initWeights() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void applyDropOut() {
		// TODO Auto-generated method stub
		
	}

	
}
