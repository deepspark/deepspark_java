package org.acl.deepspark.nn.layers.cnn;

import org.acl.deepspark.nn.layers.BaseLayer;
import org.acl.deepspark.nn.weights.WeightUtil;
import org.jblas.DoubleMatrix;
import org.jblas.SimpleBlas;
import org.jblas.ranges.RangeUtils;

public class ConvolutionLayer extends BaseLayer {
	private int dimIn, dimOut; // in/out spec.
	private DoubleMatrix[] input;
	private int imageRows, imageCols, numChannels;
	private int filterRows, filterCols, numFilters; // filter spec.
	private DoubleMatrix[][] W; // filterId, x, y
	private double[] bias;
		
	private int[] stride = {1, 1};
	private int zeroPadding = 0;
	private boolean useZeroPadding = true;

	public ConvolutionLayer(DoubleMatrix input, int filterRows, int filterCols, int numFilters) {
		super();
		this.input = new DoubleMatrix[1];
		this.input[0] = input;
		
		this.imageRows = input.rows;
		this.imageCols = input.columns;
		this.numChannels = 1;
		
		this.filterRows = filterRows;
		this.filterCols = filterCols;
		this.numFilters = numFilters;
		
		W = new DoubleMatrix[numFilters][numChannels];
		bias = new double[numFilters];
		
		for(int i = 0; i < numFilters; i++) {
			for(int j = 0; j < numChannels; j++) {
				W[i][j] = WeightUtil.randInitWeights(filterRows, filterCols);
			}
			bias[i] = 0;
		}
	}
	
	public ConvolutionLayer(DoubleMatrix[] input, int filterRows, int filterCols, int numFilters) {
		super();
		this.input = input;
		
		this.imageRows = input[0].rows;
		this.imageCols = input[0].columns;
		this.numChannels = input.length;
		
		this.filterRows = filterRows;
		this.filterCols = filterCols;
		this.numFilters = numFilters;
		
		W = new DoubleMatrix[numFilters][numChannels];
		bias = new double[numFilters];
		
		for(int i = 0; i < numFilters; i++) {
			for(int j = 0; j < numChannels; j++) {
				W[i][j] = WeightUtil.randInitWeights(filterRows, filterCols);
			}
			bias[i] = 0;
		}
	}
	
	public ConvolutionLayer(int row, int col, int channel, int filterRows, int filterCols, int numFilters) {
		super();
		
		this.imageRows = row;
		this.imageCols = col;
		this.numChannels = channel;
		
		this.filterRows = filterRows;
		this.filterCols = filterCols;
		this.numFilters = numFilters;
		
		W = new DoubleMatrix[numFilters][channel];
		bias = new double[numFilters];
		
		for(int i = 0; i < numFilters; i++) {
			for(int j = 0; j < numChannels; j++) {
				W[i][j] = WeightUtil.randInitWeights(filterRows, filterCols);
			}
			bias[i] = 0;
		}
	}
	
	public void setFilterWeights(DoubleMatrix[][] filters) {
		W = filters;
	}
	
	public DoubleMatrix[][] getFilterWeights() {
		return W;
	}

	public int getNumOfFilter() {
		return numFilters;
	}

	public DoubleMatrix[] convolution(DoubleMatrix filter) {
		DoubleMatrix[] data = new DoubleMatrix[numFilters];
		DoubleMatrix temp = new DoubleMatrix(imageRows - filterRows + 1, imageCols - filterCols + 1);;
		
		for(int i = 0; i < numFilters; i++) {
			data[i] = new DoubleMatrix(imageRows - filterRows + 1, imageCols -filterCols + 1);
			for(int j = 0; j < numChannels; j++) {
				//flip for 2d-convolution
				for(int k = 0; k < filter.rows / 2 ; k++)
					filter.swapRows(k, filter.rows - 1 - k);
				for(int k = 0; k < filter.columns / 2 ; k++)
					filter.swapColumns(k, filter.columns - 1 - k);
				
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
	
	public DoubleMatrix[] convolution(DoubleMatrix[] filters) {
		int numFilters = filters.length;
		DoubleMatrix[] data = new DoubleMatrix[numFilters];
		DoubleMatrix filter;
		DoubleMatrix temp = new DoubleMatrix(imageRows - filterRows + 1, imageCols -filterCols + 1);;
		
		// check: dims(image) > dims(filter)
		
		for(int i = 0; i < numFilters; i++) {
			data[i] = new DoubleMatrix(imageRows - filterRows + 1, imageCols -filterCols + 1);
			for(int j = 0; j < numChannels; j++) {
				filter = new DoubleMatrix(filters[i].toArray2());
				
				//flip for 2d-convolution
				for(int k = 0; k < filter.rows / 2 ; k++)
					filter.swapRows(k, filter.rows - 1 - k);
				for(int k = 0; k < filter.columns / 2 ; k++)
					filter.swapColumns(k, filter.columns - 1 - k);
				
				temp.fill(0.0);
				// calculate convolution
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
	
	// convolution for multiple channel input
	public DoubleMatrix[] convolution() {
		DoubleMatrix[] data = new DoubleMatrix[numFilters];
		DoubleMatrix filter;
		DoubleMatrix temp = new DoubleMatrix(imageRows - filterRows + 1, imageCols - filterCols + 1);
		
		// check: dims(image) > dims(filter)

		for(int i = 0; i < numFilters; i++) {
			data[i] = new DoubleMatrix(imageRows - filterRows + 1, imageCols - filterCols + 1);
			for(int j = 0; j < numChannels; j++) {
				filter = new DoubleMatrix(W[i][j].toArray2());
				
				//flip for 2d-convolution
				for(int k = 0; k < filter.rows / 2 ; k++)
					filter.swapRows(k, filter.rows - 1 - k);
				for(int k = 0; k < filter.columns / 2 ; k++)
					filter.swapColumns(k, filter.columns - 1 - k);
				
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
		if (outputDelta == null || outputDelta.length <=0 ) {
			
			DoubleMatrix[] inputDelta = new DoubleMatrix[numChannels];
			DoubleMatrix temp = new DoubleMatrix(imageRows, imageCols);
			DoubleMatrix filter;
			
			// check: dims(image) > dims(filter)
			int conv = 0;
			for (int j = 0; j < numChannels; j++) {
				inputDelta[j] = new DoubleMatrix(imageRows, imageCols);
				for (int i = 0; i < numFilters; i++) {
					filter = new DoubleMatrix(W[i][j].toArray2());
					
					temp.fill(0.0);
					// calculate convolution
					for (int r = 0; r < imageRows; r++) {
						for (int c = 0; c < imageCols ; c++) {
							for (int m = 0; m < filterRows; m++) {
								for (int n = 0; n < filterCols; n++) {
									if (r-m < 0 || c-n < 0)
										continue;
									conv += outputDelta[i].get(r-m, c-n) * filter.get(m,n);
								}
							}
							conv = 0;
							temp.put(r, c, conv);
						}
					}
					inputDelta[j].addi(temp);
				}
			}
			return inputDelta;
		}
		return null;
	}
}
