package org.acl.deepspark.utils;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

import org.jblas.DoubleMatrix;

public class MnistLoader {
	
	private static final int dimRows = 28;
	private static final int dimLabel = 10;
	
	public static DoubleMatrix[] loadData(String path) {
		BufferedReader reader = null;
		ArrayList<DoubleMatrix> data = new ArrayList<DoubleMatrix>();
		double[] featureVec = new double[dimRows * dimRows];
		try {
			reader = new BufferedReader(new FileReader(path));
			String line = null;
			String[] feature = null;
			while((line = reader.readLine()) != null) {
				feature = line.split("\t");
				for(int i = 0; i < feature.length -1;i++)
					featureVec[i] = Double.parseDouble(feature[i]);
				data.add(new DoubleMatrix(dimRows, dimRows, featureVec.clone()).transpose());			
			}
		} catch(IOException e) {
			e.printStackTrace();
		} finally {
			if (reader != null) {
				try {
					reader.close();
				} catch(IOException e) {}
			}
		}
		
		DoubleMatrix[] images = new DoubleMatrix[data.size()];
		int size = data.size();
		for(int i = 0 ; i < size; i++)
			images[i] = data.get(i);
		return images;
	}
	
	public static DoubleMatrix[] loadLabel(String path) {
		BufferedReader reader = null;
		ArrayList<DoubleMatrix> data = new ArrayList<DoubleMatrix>();
		double[] labelVec = new double[dimLabel];
		
		try {
			reader = new BufferedReader(new FileReader(path));
			double label;
			String line = null;
			String[] feature = null;
			while((line = reader.readLine()) != null) {
				feature = line.split("\t");
				label = Double.parseDouble(feature[dimRows * dimRows]);
				for(int i = 0; i < dimLabel; i++) {
					labelVec[i] = (label == i) ?  1.0 : 0.0;
				}
				data.add(new DoubleMatrix(labelVec.clone()));			
			}
		} catch(IOException e) {
			e.printStackTrace();
		} finally {
			if (reader != null) {
				try {
					reader.close();
				} catch(IOException e) {}
			}
		}
		
		DoubleMatrix[] images = new DoubleMatrix[data.size()];
		int size = data.size();
		for(int i = 0 ; i < size; i++)
			images[i] = data.get(i);
		return images;
	}
	
}
