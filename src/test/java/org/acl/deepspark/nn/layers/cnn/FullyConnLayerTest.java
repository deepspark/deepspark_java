package org.acl.deepspark.nn.layers.cnn;

import org.acl.deepspark.nn.layers.FullyConnLayer;
import org.acl.deepspark.nn.weights.WeightUtil;
import org.jblas.DoubleMatrix;

public class FullyConnLayerTest {
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
		
		FullyConnLayer fullyConnLayer = new FullyConnLayer(inputArr, 10);		
		System.out.println(WeightUtil.flat2Vec(inputArr));
		
		System.out.println(fullyConnLayer.getWeight());
		
		
		DoubleMatrix[] output = fullyConnLayer.getOutput();
		for(DoubleMatrix matrix : output)
			System.out.println(matrix);
		
		System.out.println(input1);
		DoubleMatrix[] inputDelta = fullyConnLayer.update(output);
		for(DoubleMatrix matrix : inputDelta)
			System.out.println(matrix);
		
		
		System.out.println((input1.mul(input1.mul(-1.0).add(1.0))));
		
		/** FullyConn feedforward complete **/
		
	}
}
