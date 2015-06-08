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
	private DoubleMatrix[] output;
		
	private int[] stride = {1, 1};
	private int zeroPadding = 0;
	private boolean useZeroPadding = true;

	
	/** Modified **/
	public ConvolutionLayer(int filterRows, int filterCols, int numFilters) {
		super();
		this.filterRows = filterRows;
		this.filterCols = filterCols;
		this.numFilters = numFilters;
		
		//initWeights();
	}
	
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
			bias[i] = 0.01;
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
		/** Modified **/
		if(W == null)
			initWeights();
		
		
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
		output = activate(convolution());
		return output;
	}

	@Override
	public DoubleMatrix[] update(DoubleMatrix[] propDelta) {
		// TODO Auto-generated method stub
		DoubleMatrix[] delta = new DoubleMatrix[propDelta.length];
		for(int i = 0 ; i < propDelta.length; i++)
			propDelta[i].muli(output[i].mul(output[i].mul(-1.0).add(1.0)));
			//delta[i] = propDelta[i].mul(output[i].mul(output[i].mul(-1.0).add(1.0)));
		
		DoubleMatrix deltaWeight = new DoubleMatrix(filterRows, filterCols);
		// update Weights
		for (int i = 0; i < numFilters; i++) {
			for (int j = 0; j < numChannels; j++) {
				deltaWeight.fill(0.0);
				for(int r = 0; r < filterRows; r++) {
					for(int c = 0; c < filterCols ; c++) {
						deltaWeight.put(r, c, SimpleBlas.dot
								(input[j].get(RangeUtils.interval(r, r + propDelta[i].rows),
										   	  RangeUtils.interval(c, c + propDelta[i].columns)), propDelta[i]));
					}
				}
				W[i][j].subi(deltaWeight.mul(learningRate));
			}
		}
		// return inputLayer delta
		return deriveDelta (propDelta);
	}
	
	public DoubleMatrix[] deriveDelta(DoubleMatrix[] propDelta) {
		if (propDelta == null || propDelta.length <= 0)
			return null;
		
		DoubleMatrix[] delta = new DoubleMatrix[numChannels];
		DoubleMatrix temp = new DoubleMatrix(dimRows, dimCols);
		DoubleMatrix filter;
		
		// check: dims(image) > dims(filter)
		int conv;
		for (int j = 0; j < numChannels; j++) {
			delta[j] = new DoubleMatrix(dimRows, dimCols);
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
								conv += propDelta[i].get(r-m, c-n) * filter.get(m,n);
							}
						}
						temp.put(r, c, conv);
					}
				}
				delta[j].addi(temp);
			}
		}
		return delta;
	}

	

	@Override
	public void applyDropOut() {
		// TODO Auto-generated method stub
		
	}
}
