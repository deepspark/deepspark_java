package org.acl.deepspark.nn.layers.cnn;
import org.acl.deepspark.nn.layers.cnn.ConvolutionLayer;
import org.jblas.DoubleMatrix;


public class ConvolutionLayerTest {
	public static void main(String[] args) {
		double[][] a = {{1,3,5,7,9,11}, {13,11,9,7,5,3}, {10,6,8,4,2,1}, {9,7,5,3,1,3}, {14,12,10,8,6,4}, {16,14,7,9,8,3}};
		double[][] b = {{1,2,4,3,-1,0}, {2,4,3,5,7,-3}, {-2,1,-4,2,0,-3}, {0,-1,-2,2,3,-4}, {1,2,-3,-2,1,-1}, {3,2,1,-1,-2,3}};
		double[][] c = {{1,3,5,7,9,11}, {13,11,9,7,5,3}, {10,6,8,4,2,1}, {9,7,5,3,1,3}, {14,12,10,8,6,4}, {16,14,7,9,8,3}};
		double[][] d = {{1,2,4,3,-1,0}, {2,4,3,5,7,-3}, {-2,1,-4,2,0,-3}, {0,-1,-2,2,3,-4}, {1,2,-3,-2,1,-1}, {3,2,1,-1,-2,3}};
		
		double[][] filter1_1 = {{1,2,1}, {0,0,0}, {-1,-2,-1}};
		double[][] filter1_2 = {{-1,-2,3}, {-2,-1,0}, {0,1,-1}};
		double[][] filter1_3 = {{-1,-2,3}, {-2,-1,0}, {0,1,-1}};
		double[][] filter1_4 = {{-1,-2,3}, {-2,-1,0}, {0,1,-1}};
		
		double[][] filter2_1 = {{1,2,1}, {0,0,0}, {-1,-2,-1}};
		double[][] filter2_2 = {{-1,-2,3}, {-2,-1,0}, {0,1,-1}};
		double[][] filter2_3 = {{-1,-2,3}, {-2,-1,0}, {0,1,-1}};
		double[][] filter2_4 = {{-1,-2,3}, {-2,-1,0}, {0,1,-1}};
		
		DoubleMatrix input1 = new DoubleMatrix(a);
		DoubleMatrix input2 = new DoubleMatrix(b);
		DoubleMatrix input3 = new DoubleMatrix(c);
		DoubleMatrix input4 = new DoubleMatrix(d);
		DoubleMatrix[] inputArr = {input1, input2, input3, input4};
		
//		DoubleMatrix filter = new DoubleMatrix(d);
//		DoubleMatrix filter2 = new DoubleMatrix(e);
//		DoubleMatrix[] filterArr = {filter, filter2};
		
		ConvolutionLayer convLayer = new ConvolutionLayer(inputArr, 3, 3, 2);
		DoubleMatrix[] result = convLayer.convolution();
		
		System.out.println("Convolution Filters");
		for (int i = 0; i < convLayer.getNumOfFilter(); i++) {
			for (int j = 0; j < convLayer.getNumOfChannels(); j++) {
				System.out.println(convLayer.getFilterWeights()[i][j]);
			}
		}
		
		System.out.println("Convolution Result");
		for (DoubleMatrix matrix : result) {
			System.out.println(matrix.toString());
		}
	}
}
