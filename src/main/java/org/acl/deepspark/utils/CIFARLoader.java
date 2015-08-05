package org.acl.deepspark.utils;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;

import org.acl.deepspark.data.Sample;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class CIFARLoader implements Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 2453031569679527445L;
		
	private static final int dimRows = 32;
	private static final int channel = 3;
	private static final int dimLabel = 10;
			
	public static Sample[] loadIntoSamples(String path, boolean normalize) throws IOException {
		int label;
		byte[] data = new byte[dimRows * dimRows];
		ArrayList<Sample> samples = new ArrayList<Sample>();
		
		final int[] sampleDim = new int[]{channel, dimRows, dimRows};
		
		FileInputStream in = new FileInputStream(path);
		
		while((label = in.read()) != -1) {
			double[] labelVec = new double[dimLabel];
			double[][] featureVec = new double[channel][];
			for(int i = 0; i < dimLabel; i++) {
				labelVec[i] = (label == i) ?  1.0 : 0.0;
			}
			
			Sample s = new Sample();
			s.data = Nd4j.create(sampleDim);
			s.label = Nd4j.create(labelVec);
			
			for(int i = 0;i < channel; i++) {
				in.read(data);
				featureVec[i] = new double[dimRows * dimRows];
				for(int j = 0; j < featureVec.length; j++) {
					featureVec[i][j] = (double) data[j];
				}
				
				INDArray channelData = Nd4j.create(featureVec[i]).reshape(dimRows,dimRows).transpose();
				s.data.putSlice(i, channelData);				
			}				
			
			samples.add(s);
		}
		
		in.close();
		
		Sample[] arr = new Sample[samples.size()];
		arr = samples.toArray(arr);

		System.out.println(String.format("Loaded %d samples from %s", samples.size(), path));
		return arr;
	}
}