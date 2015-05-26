package org.acl.deepspark.nn.layers.cnn;

import org.acl.deepspark.nn.layers.BaseLayer;
import org.acl.deepspark.nn.weights.WeightUtil;
import org.jblas.DoubleMatrix;
import org.jblas.SimpleBlas;
import org.jblas.ranges.RangeUtils;

public class ConvolutionLayer extends BaseLayer {
	private int filterRows, filterCols, numFilters; // filter spec.
	private DoubleMatrix[][] W; // filterId, x, y
	private double[] bias;
		
	private int[] stride = {1, 1};
	private int zeroPadding = 0;
	private boolean useZeroPadding = true;

	public ConvolutionLayer(DoubleMatrix input, int filterRows, int filterCols, int numFilters) {
		super(input);
		this.filterRows = filterRows;
		this.filterCols = filterCols;
		this.numFilters = numFilters;
		
		initWeights();
	}
	
	public ConvolutionLayer(DoubleMatrix[] input, int filterRows, int filterCols, int numFilters) {
		super(input);
		this.filterRows = filterRows;
		this.filterCols = filterCols;
		this.numFilters = numFilters;
		
		initWeights();
	}
	
	
	public void setFilterWeights(DoubleMatrix[][] filters) {
		W = filters;
	}
	
	public DoubleMatrix[][] getFilterWeights() {
		return W;
	}
	
	@Override
	public void initWeights() {
		W = new DoubleMatrix[numFilters][numChannels];
		bias = new double[numFilters];
		
		for(int i = 0; i < numFilters; i++) {
			for(int j = 0; j < numChannels; j++) {
				W[i][j] = WeightUtil.randInitWeights(filterRows, filterCols);
			}
			bias[i] = 0;
		}
	}

	public int getNumOfChannels() {
		return numChannels;
	}
	
	public int getNumOfFilter() {
		return numFilters;
	}
	
	private int getOutputRows() {
		return  dimRows - filterRows + 1;
	}
	
	private int getOutputCols() {
		return  dimCols - filterCols + 1;
	}
	
	
	// convolution for multiple channel input
	public DoubleMatrix[] convolution() {
		DoubleMatrix[] data = new DoubleMatrix[numFilters];
		DoubleMatrix filter;
		DoubleMatrix temp = new DoubleMatrix(dimRows-filterRows+1, dimCols-filterCols+1);
		
		// check: dims(image) > dims(filter)

		for(int i = 0; i < numFilters; i++) {
			data[i] = new DoubleMatrix(dimRows - filterRows + 1, dimCols - filterCols + 1);
			for(int j = 0; j < numChannels; j++) {
				filter = new DoubleMatrix(W[i][j].toArray2());
				
				//flip for 2d-convolution
/*				for(int k = 0; k < filter.rows / 2 ; k++)
					filter.swapRows(k, filter.rows - 1 - k);
				for(int k = 0; k < filter.columns / 2 ; k++)
					filter.swapColumns(k, filter.columns - 1 - k);
*/				
				temp.fill(0.0);
				// calculate convolutions
				for(int r = 0; r < temp.rows; r++) {
					for(int c = 0; c < temp.columns ; c++) {
						temp.put(r, c, SimpleBlas.dot
								(input[j].get(RangeUtils.interval(r, r + filter.rows),
										   	  RangeUtils.interval(c, c + filter.columns)), filter));
					}
				}
				data[i].addi(temp);
			}
			data[i].addi(bias[i]);
		}
		return data;
	}
	
	/*
	
	public DoubleMatrix convolutionDownSample() {
		if(input != null) {
			output = pooling(activate(convolution(input)));
			
			DoubleMatrix temp = output[0]; 
			for(int i = 1; i < output.length; i++) {
				DoubleMatrix.concatVertically(temp, output[i]);
			}
			return temp;
		}
		return null;
	}
	
	
	public void feedForward() {
		outputLayer = new FullyConnLayer(convolutionDownSample(), dimOut);
		outputLayer.feedForward();
	}
	*/
	
	@Override
	public DoubleMatrix[] getOutput() {
		return activate(convolution());
	}

	@Override
	public DoubleMatrix[] update(DoubleMatrix[] outputDelta) {
		// TODO Auto-generated method stub
		DoubleMatrix deltaWeight = new DoubleMatrix(filterRows, filterCols);
		
		// update Weights
		for (int i = 0; i < numFilters; i++) {
			for (int j = 0; j < numChannels; j++) {
				deltaWeight.fill(0.0);
				for(int r = 0; r < filterRows; r++) {
					for(int c = 0; c < filterCols ; c++) {
						deltaWeight.put(r, c, SimpleBlas.dot
								(input[j].get(RangeUtils.interval(r, r + outputDelta[i].rows),
										   	  RangeUtils.interval(c, c + outputDelta[i].columns)), outputDelta[i]));
					}
				}
				W[i][j].addi(deltaWeight);
			}
		}
		// return inputLayer delta
		return deriveDelta (outputDelta);
	}
	
	public DoubleMatrix[] deriveDelta(DoubleMatrix[] outputDelta) {
		if (outputDelta == null || outputDelta.length <= 0)
			return null;
		
		DoubleMatrix[] inputDelta = new DoubleMatrix[numChannels];
		DoubleMatrix temp = new DoubleMatrix(dimRows, dimCols);
		DoubleMatrix filter;
		
		// check: dims(image) > dims(filter)
		int conv;
		for (int j = 0; j < numChannels; j++) {
			inputDelta[j] = new DoubleMatrix(dimRows, dimCols);
			for (int i = 0; i < numFilters; i++) {
				filter = new DoubleMatrix(W[i][j].toArray2());
				
				temp.fill(0.0);
				// calculate convolution
				for (int r = 0; r < dimRows; r++) {
					for (int c = 0; c < dimCols ; c++) {
						conv = 0;
						for (int m = 0; m < filterRows; m++) {
							for (int n = 0; n < filterCols; n++) {
								if (r-m < 0 || r-m >= getOutputRows() || c-n < 0 || c-n >= getOutputCols())
									continue;
								conv += outputDelta[i].get(r-m, c-n) * filter.get(m,n);
							}
						}
						temp.put(r, c, conv);
					}
				}
				inputDelta[j].addi(temp);
			}
		}
		return inputDelta;
	}

	

	@Override
	public void applyDropOut() {
		// TODO Auto-generated method stub
		
	}
}
