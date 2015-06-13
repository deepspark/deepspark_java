package org.acl.deepspark.utils;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

import org.jblas.DoubleMatrix;

public class MnistLoader {
	
	private static final int dimRows = 28;
	private static final int dimLabel = 10;
	
	private DoubleMatrix[] data;
	private DoubleMatrix[] label;
	
	public DoubleMatrix[] getData() {
		return data;
	}

	public DoubleMatrix[] getLabel() {
		return label;
	}
	
	public void load(String path, boolean normalize) {
		BufferedReader reader = null;
		double label;
		
		ArrayList<DoubleMatrix> data = new ArrayList<DoubleMatrix>();
		ArrayList<DoubleMatrix> labelList = new ArrayList<DoubleMatrix>();
		double[] featureVec = new double[dimRows * dimRows];
		try {
			reader = new BufferedReader(new FileReader(path));
			String line = null;
			String[] feature = null;
			while((line = reader.readLine()) != null) {
				feature = line.split("\t");
				label = Double.parseDouble(feature[dimRows * dimRows]);
				double[] labelVec = new double[dimLabel];		
				for(int i = 0; i < dimLabel; i++) {
					labelVec[i] = (label == i) ?  1.0 : 0.0;
				}
				labelList.add(new DoubleMatrix(labelVec));
				
				for(int i = 0; i < feature.length -1;i++)
					featureVec[i] = Double.parseDouble(feature[i]);
				
				DoubleMatrix sample = new DoubleMatrix(dimRows, dimRows, featureVec.clone()).transpose();
				if(normalize)
					data.add(sample.divi(256));
				else
					data.add(sample);
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
		
		
		this.data =  new DoubleMatrix[data.size()];
		this.data = data.toArray(this.data);
		this.label = new DoubleMatrix[labelList.size()];
		this.label = labelList.toArray(this.label);	
	}
	
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
