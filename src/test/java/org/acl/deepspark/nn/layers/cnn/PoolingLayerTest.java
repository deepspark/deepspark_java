package org.acl.deepspark.nn.layers.cnn;

import org.jblas.DoubleMatrix;
import org.acl.deepspark.nn.layers.cnn.PoolingLayer;

public class PoolingLayerTest {
	public static void main(String[] args) {
		
		double[][] input = {{1,3,5,7,9,11}, {13,11,9,7,5,3}, {10,6,8,4,2,1}, {9,7,5,3,1,3}, {14,12,10,8,6,4}, {16,14,7,9,8,3}};
		double[][] input2 = {{1,2,4,3,-1,0}, {2,4,3,5,7,-3}, {-2,1,-4,2,0,-3}, {0,-1,-2,2,3,-4}, {1,2,-3,-2,1,-1}, {3,2,1,-1,-2,3}};
		
		DoubleMatrix inputMat = new DoubleMatrix(input);
		DoubleMatrix inputMat2 = new DoubleMatrix(input2);
		DoubleMatrix[] inputMatrices = {inputMat, inputMat2};
		
		// pooling layer constructor
		PoolingLayer poolingLayer = new PoolingLayer(inputMatrices, 2);
		PoolingLayer poolingLayer2 = new PoolingLayer(inputMatrices, 3);
		
		// pooling a single image
		System.out.println(poolingLayer.pooling(inputMat).toString());
		System.out.println(poolingLayer.pooling(inputMat2).toString());
		System.out.println();
		
		System.out.println(poolingLayer2.pooling(inputMat).toString());
		System.out.println(poolingLayer2.pooling(inputMat2).toString());
		System.out.println();
		
		// overall pooling check
		DoubleMatrix[] result = poolingLayer.pooling();
		DoubleMatrix[] result2 = poolingLayer2.pooling();
		
		for(DoubleMatrix matrix : result)
			System.out.println(matrix.toString());
		for(DoubleMatrix matrix : result2)
			System.out.println(matrix.toString());
		
		// maxIndices check
		int[][] layer1 = poolingLayer.maxIndices;
		int[][] layer2 = poolingLayer2.maxIndices;
		
		System.out.println("poolingLayer1 maxIdices:");
		for(int idx : layer1[0])
			System.out.println(String.valueOf(idx));
		for(int idx : layer1[1])
			System.out.println(String.valueOf(idx));
		System.out.println("poolingLayer2 maxIdices:");
		for(int idx : layer2[0])
			System.out.println(String.valueOf(idx));
		for(int idx : layer2[1])
			System.out.println(String.valueOf(idx));
		
		
		// update 
		poolingLayer.setDelta(result);
		DoubleMatrix[] update1 = poolingLayer.deriveDelta();
		System.out.println("poolingLayer1 update check");
		for(DoubleMatrix matrix : update1)
			System.out.println(matrix);
		
		poolingLayer2.setDelta(result2);
		DoubleMatrix[] update2 = poolingLayer2.deriveDelta();
		System.out.println("poolingLayer2 update check");
		for(DoubleMatrix matrix : update2)
			System.out.println(matrix);
		
		
		
		/*** feedforward test complete ***/
		/*** update test complete ***/
	}
	
}
