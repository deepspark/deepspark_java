package org.acl.deepspark.nn.layers.cnn;
import org.acl.deepspark.nn.layers.cnn.ConvolutionLayer;
import org.jblas.DoubleMatrix;


public class ConvolutionLayerTest {
	
	// FeedForward Test
	public static void main(String[] args) {
		double[][] a = {{1,3,5,7,9,11}, {13,11,9,7,5,3}, {10,6,8,4,2,1}, {9,7,5,3,1,3}, {14,12,10,8,6,4}, {16,14,7,9,8,3}};
		double[][] b = {{1,2,4,3,-1,0}, {2,4,3,5,7,-3}, {-2,1,-4,2,0,-3}, {0,-1,-2,2,3,-4}, {1,2,-3,-2,1,-1}, {3,2,1,-1,-2,3}};
		double[][] c = {{2,3,5,7,9,6}, {1,11,3,7,5,3}, {10,6,4,4,2,1}, {9,7,6,3,1,3}, {8,12,10,8,1,4}, {16,2,7,9,8,3}};
		double[][] d = {{0,2,4,3,-1,0}, {2,2,3,5,7,-3}, {-2,1,-4,3,0,-3}, {0,-1,-2,2,3,-4}, {4,2,-3,-2,1,-1}, {3,2,4,-1,-2,3}};

		DoubleMatrix input1 = new DoubleMatrix(a);
		DoubleMatrix input2 = new DoubleMatrix(b);
		DoubleMatrix input3 = new DoubleMatrix(c);
		DoubleMatrix input4 = new DoubleMatrix(d);
		DoubleMatrix[] inputArr = {input1, input2, input3, input4};

		ConvolutionLayer convLayer = new ConvolutionLayer(3, 3, 2);
		convLayer.setInput(inputArr);
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
		
/*		
		double[][] filterArr = {{2,3,5}, {1,11,3}, {10,6,4}};
		DoubleMatrix filter = new DoubleMatrix(filterArr);
		int dimRows = 8;
		int dimCols = 8;
		DoubleMatrix temp = new DoubleMatrix(dimRows, dimCols);
		
		
		System.out.println(filter);
		
		temp.fill(0.0);
		double conv = 0;
		// calculate convolution
		for (int r = 0; r < dimRows; r++) {
			for (int cc = 0; cc < dimCols ; cc++) {
				conv = 0;
				for (int m = 0; m < 3; m++) {
					for (int n = 0; n < 3; n++) {
						if (r-m < 0 || r-m >= 6 || cc-n < 0 || cc-n >= 6)
							continue;
						conv += input1.get(r-m, cc-n) * filter.get(m,n);
					}
				}
				temp.put(r, cc, conv);
			}
		}
		System.out.println("derive Delta Result");
		System.out.println(temp);*/
	}
		
		/** feedforward test complete **/
	
}
