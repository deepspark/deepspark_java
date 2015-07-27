package org.acl.deepspark.utils;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Serializable;
import java.util.ArrayList;

import org.acl.deepspark.data.Sample;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.jblas.DoubleMatrix;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.factory.Nd4j;

public class MnistLoader implements Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 4845357475294611873L;
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
	
	public static Sample[] loadFromHDFS(String path, boolean normalize) {
		BufferedReader reader = null;
		double label;
		ArrayList<Sample> samples = new ArrayList<Sample>();
		
		double[] featureVec = new double[dimRows * dimRows];
		try {
			Path p = new Path(path);
			FileSystem fs = FileSystem.get(new Configuration());
			reader = new BufferedReader(new InputStreamReader(fs.open(p)));
			String line = null;
			String[] feature = null;
			while((line = reader.readLine()) != null) {
				feature = line.split("\t");
				label = Double.parseDouble(feature[dimRows * dimRows]);
				double[] labelVec = new double[dimLabel];		
				for(int i = 0; i < dimLabel; i++) {
					labelVec[i] = (label == i) ?  1.0 : 0.0;
				}
				DoubleMatrix sampleLabel = new DoubleMatrix(labelVec);
				
				for(int i = 0; i < feature.length -1;i++)
					featureVec[i] = Double.parseDouble(feature[i]);
				
				Sample s = new Sample();
				DoubleMatrix[] sample = new DoubleMatrix[1];
				sample[0] = new DoubleMatrix(dimRows, dimRows, featureVec.clone()).transpose();
				
				if(normalize)
					sample[0].divi(256);
				
				//s.data = sample;
				//s.label = sampleLabel;
				
				samples.add(s);
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
		
		Sample[] arr = new Sample[samples.size()];
		arr = samples.toArray(arr);
		
		return arr;
	}
	
	public static Sample[] loadIntoSamples(String path, boolean normalize) {
		BufferedReader reader = null;
		double label;
		ArrayList<Sample> samples = new ArrayList<Sample>();
		
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
				DoubleMatrix sampleLabel = new DoubleMatrix(labelVec);
				
				for(int i = 0; i < feature.length -1;i++)
					featureVec[i] = Double.parseDouble(feature[i]);
				
				Sample s = new Sample();
				DoubleMatrix[] sample = new DoubleMatrix[1];
				sample[0] = new DoubleMatrix(dimRows, dimRows, featureVec.clone()).transpose();
				
				if(normalize)
					sample[0].divi(256);
				int[] dimData = {1, dimRows, dimRows};
				
				s.data = Nd4j.zeros(dimData);
				s.data.setData(Nd4j.createBuffer(featureVec));
				s.label = Nd4j.zeros(dimLabel,1);
				s.label.setData(Nd4j.createBuffer(labelVec));;
				
				samples.add(s);
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
		
		Sample[] arr = new Sample[samples.size()];
		arr = samples.toArray(arr);

		System.out.println(String.format("Loaded %d samples from %s", samples.size(), path));
		return arr;
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
