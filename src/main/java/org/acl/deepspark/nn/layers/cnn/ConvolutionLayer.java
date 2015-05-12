package org.acl.deepspark.nn.layers.cnn;

import org.acl.deepspark.nn.layers.BaseLayer;
import org.acl.deepspark.nn.layers.HiddenLayer;
import org.jblas.DoubleMatrix;
import org.jblas.SimpleBlas;
import org.jblas.ranges.RangeUtils;

public class ConvolutionLayer extends BaseLayer{
	private int dimIn, dimOut; // in/out spec.
	private DoubleMatrix[] input;
	private DoubleMatrix[] output;
	private int imageRows, imageCols, numChannel;
	private int filterRows, filterCols, numFilters; // filter spec.
	private DoubleMatrix[] W; // filterId, x, y
	private double[] bias;
	
	private HiddenLayer outputLayer;
	
	private int numIteration;
	private int batchSize;
	private int[] stride;
	private int[] poolSize;
	private int zeroPadding;
	
	public ConvolutionLayer(double[][] input) {
		
	}
	
	
	public ConvolutionLayer(DoubleMatrix[] input, int filterRows, int filterCols, int numFilters) {
		super();
		this.input = input;
		
		this.imageRows = input[0].rows;
		this.imageCols = input[0].columns;
		this.numFilters = input.length;
		
		this.filterRows = filterRows;
		this.filterCols = filterCols;
		this.numFilters = numFilters;
		
		W = new DoubleMatrix[numFilters];
		bias = new double[numFilters];
		
		for(int i = 0; i < numFilters; i++) {
			W[i] = randInitWeights(filterRows, filterCols);
			bias[i] = 0;
		}
	}
	
	public ConvolutionLayer(int row, int col, int channel, int filterRows, int filterCols, int numFilters) {
		super();
		
		this.imageRows = row;
		this.imageCols = col;
		this.numChannel = channel;
		
		this.filterRows = filterRows;
		this.filterCols = filterCols;
		this.numFilters = numFilters;
		
		W = new DoubleMatrix[numFilters];
		bias = new double[numFilters];
		
		for(int i = 0; i < numFilters; i++) {
			W[i] = randInitWeights(filterRows, filterCols);
			bias[i] = 0;
		}
	}
	
	/* TODO: apply initialization method */
	private DoubleMatrix randInitWeights(int dimRow, int dimCol) {
		return DoubleMatrix.rand(dimRow, dimCol);
	}

	public int getNumOfFilter() {
		return numFilters;
	}
	
	public DoubleMatrix[] convolution(DoubleMatrix image) {
		DoubleMatrix[] data = new DoubleMatrix[numFilters];
		DoubleMatrix filter;
		for(int i = 0; i < numFilters; i++) {
			data[i] = new DoubleMatrix(image.rows - filterRows + 1, image.columns -filterCols + 1);
			filter = new DoubleMatrix(W[i].toArray2());
			
			//flip for 2d-convolution
			for(int k = 0; k < filter.rows / 2 ; k++)
				filter.swapRows(k, filter.rows - 1 - k);
			for(int k = 0; k < filter.columns / 2 ; k++)
				filter.swapColumns(k, filter.columns - 1 - k);
			
			// calculate convolutions
			for(int r = 0; r < data[i].rows; r++) {
				for(int c = 0; c < data[i].columns ; c++) {
					data[i].put(r, c, SimpleBlas.dot
							(image.get(RangeUtils.interval(r, r + filter.rows),
									   RangeUtils.interval(c, c + filter.columns)), filter));
				}
			}
			data[i].addi(bias[i]);
		}
		return data;
	}
	
	// convolution for multiple channel input
	public DoubleMatrix[] convolution(DoubleMatrix[] image) {
		DoubleMatrix[] data = new DoubleMatrix[numFilters];
		DoubleMatrix filter;
		DoubleMatrix temp;
		
		// check: dims(image) > dims(filter)
		
		for(int i = 0; i < numFilters; i++) {
			data[i] = new DoubleMatrix(image[0].rows - filterRows + 1, image[0].columns -filterCols + 1);
			for(int n = 0; n < numChannel; n++) {
				temp = new DoubleMatrix(image[0].rows - filterRows + 1, image[0].columns -filterCols + 1);
				filter = new DoubleMatrix(W[i].toArray2());
				
				//flip for 2d-convolution
				for(int k = 0; k < filter.rows / 2 ; k++)
					filter.swapRows(k, filter.rows - 1 - k);
				for(int k = 0; k < filter.columns / 2 ; k++)
					filter.swapColumns(k, filter.columns - 1 - k);
				
				// calculate convolutions
				for(int r = 0; r < temp.rows; r++) {
					for(int c = 0; c < temp.columns ; c++) {
						temp.put(r, c, SimpleBlas.dot
								(image[n].get(RangeUtils.interval(r, r + filter.rows),
										   	  RangeUtils.interval(c, c + filter.columns)), filter));
					}
				}
				data[i].add(temp);
			}
			data[i].addi(bias[i]);
		}
		return data;
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
	
	public DoubleMatrix feedForward() {
		outputLayer = new HiddenLayer(convolutionDownSample(), dimOut);
		return outputLayer.feedForward();
	}
	
	@Override
	public double[] getOutput() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void update(DoubleMatrix[] weights) {
		// TODO Auto-generated method stub
		
	}
	
	public static void main(String[] args) {
		double[][] s = {{1,2,3}, {4,5,6}, {7,8,9}};
		double[][] d = {{1,2,1}, {0,0,0}, {-1,-2,-1}};
		DoubleMatrix a = new DoubleMatrix(s);
		DoubleMatrix b = new DoubleMatrix(d);
		
		System.out.println(a.get(0,0));
		
		System.out.println(a.get(1, 2));
		System.out.println(SimpleBlas.dot(a, b));
	}
}
