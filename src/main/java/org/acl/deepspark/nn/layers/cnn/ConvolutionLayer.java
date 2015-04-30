package org.acl.deepspark.nn.layers.cnn;

import org.acl.deepspark.nn.layers.BaseLayer;
import org.jblas.DoubleMatrix;
import org.jblas.SimpleBlas;
import org.jblas.ranges.IntervalRange;
import org.jblas.ranges.RangeUtils;

public class ConvolutionLayer extends BaseLayer{
	private int[] dimIn, dimOut; // in/out spec.
	private int dimRow,dimCol,numFilter; // filter spec.
	private DoubleMatrix[] W; // filterId, x, y
	private double[] bias;
	
	public ConvolutionLayer(int row, int col, int channel, int dimRow, int dimCol, int numFilter) {
		super();
		this.dimRow = dimRow;
		this.dimRow = dimCol;
		this.numFilter = numFilter;
		
		W = new DoubleMatrix[numFilter];
		bias = new double[numFilter];
		
		for(int i = 0; i < numFilter; i++) {
			W[i] = DoubleMatrix.randn(dimRow, dimCol).muli(0.1);
			bias[i] = 0;
		}
	}

	public int getNumOfFilter() {
		return numFilter;
	}
	
	public DoubleMatrix[] convolution(DoubleMatrix image) {
		DoubleMatrix[] data = new DoubleMatrix[numFilter];
		
		for(int i = 0; i < numFilter; i++) {
			data[i] = new DoubleMatrix(image.rows - dimRow + 1, image.columns -dimCol + 1);
			DoubleMatrix filter = new DoubleMatrix(W[i].toArray2());
			
			//flip for 2d-convolution
			for(int k = 0; k < filter.getRows() / 2 ; k++)
				filter.swapRows(k, filter.getRows() - 1 -k);
			for(int k = 0; k < filter.getColumns() / 2 ; k++)
				filter.swapRows(k, filter.getRows() - 1 -k);
			
			// calculate convoltions
			for(int r = 0; r < data[i].rows; r++) {
				for(int c = 0; c < data[i].columns ; c++) {
					data[i].put(r, c,
							SimpleBlas.dot(image.get(RangeUtils.interval(r, r + filter.getRows()),
							RangeUtils.interval(c, c + filter.getColumns())), filter));
				}
			}
			data[i].addi(bias[i]);
		}
			
		
		return data;
	}
	
	@Override
	public double[] getOutput() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void update(double[] weights) {
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
