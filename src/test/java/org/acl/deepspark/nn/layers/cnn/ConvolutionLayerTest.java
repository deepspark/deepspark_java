package org.acl.deepspark.nn.layers.cnn;
import org.acl.deepspark.nn.layers.cnn.ConvolutionLayer;
import org.jblas.DoubleMatrix;


public class ConvolutionLayerTest {
	public static void main(String[] args) {
		double[][] s = {{1,3,5,7,9,11}, {13,11,9,7,5,3}, {10,6,8,4,2,1}, {9,7,5,3,1,3}, {14,12,10,8,6,4}, {16,14,7,9,8,3}};
		double[][] i = {{1,2,4,3,-1,0}, {2,4,3,5,7,-3}, {-2,1,-4,2,0,-3}, {0,-1,-2,2,3,-4}, {1,2,-3,-2,1,-1}, {3,2,1,-1,-2,3}};
		double[][] d = {{1,2,1}, {0,0,0}, {-1,-2,-1}};
		double[][] e = {{-1,-2,3}, {-2,-1,0}, {0,1,-1}};
		
		DoubleMatrix input = new DoubleMatrix(s);
		DoubleMatrix input2 = new DoubleMatrix(i);
		DoubleMatrix[] inputArr = {input, input2};
		
		DoubleMatrix filter = new DoubleMatrix(d);
		DoubleMatrix filter2 = new DoubleMatrix(e);
		DoubleMatrix[] filterArr = {filter, filter2};
		
		ConvolutionLayer convLayer = new ConvolutionLayer(inputArr, 3, 3, 2);
		DoubleMatrix[] result = convLayer.convolution(filterArr);
		
		for(DoubleMatrix matrix : result) {
			System.out.println(matrix.toString());
		}
	}
}
