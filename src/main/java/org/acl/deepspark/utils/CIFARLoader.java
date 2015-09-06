package org.acl.deepspark.utils;

import java.io.*;
import java.util.ArrayList;

import org.acl.deepspark.data.Sample;
import org.acl.deepspark.data.Tensor;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
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
			
	public static Sample[] loadIntoSamples(String path, boolean normalize) {
		int label;
		byte[] data = new byte[dimRows * dimRows];
		final int[] sampleDim = new int[]{channel, dimRows, dimRows};
		ArrayList<Sample> samples = new ArrayList<Sample>();

		FileInputStream in = null;
		try {
			in = new FileInputStream(path);
			while ((label = in.read()) != -1) {
				double[] labelVec = new double[dimLabel];
				double[][] featureVec = new double[channel][];
				for (int i = 0; i < dimLabel; i++) {
					labelVec[i] = (label == i) ? 1.0 : 0.0;
				}

				Sample s = new Sample();
				s.data = Tensor.zeros(sampleDim);
				s.label = Tensor.zeros(labelVec);

				int value;
				for (int i = 0; i < channel; i++) {
					in.read(data);
					featureVec[i] = new double[dimRows * dimRows];
					for (int j = 0; j < featureVec[i].length; j++) {
						value = (int) data[j]&0xff;
						if (normalize)
							featureVec[i][j] = (double) (value - 127) / 128.0;
						else
							featureVec[i][j] = (double) value;
					}
					INDArray channelData = Nd4j.create(featureVec[i]).reshape(dimRows, dimRows);
					s.data.putSlice(i, channelData);
				}
				samples.add(s);
			}
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (in != null) {
				try {
					in.close();
				} catch (IOException e) {}
			}
		}

		Sample[] arr = new Sample[samples.size()];
		arr = samples.toArray(arr);

		System.out.println(String.format("Loaded %d samples from %s", samples.size(), path));
		return arr;
	}

	public static Sample[] loadFromHDFS(String path, boolean normalize) {
		int label;
		byte[] data = new byte[dimRows * dimRows];
		final int[] sampleDim = new int[]{channel, dimRows, dimRows};
		ArrayList<Sample> samples = new ArrayList<Sample>();

		FSDataInputStream in = null;
		try {
			Path p = new Path(path);
			FileSystem fs = FileSystem.get(new Configuration());
			in = fs.open(p);

			while ((label = in.read()) != -1) {
				double[] labelVec = new double[dimLabel];
				double[][] featureVec = new double[channel][];
				for (int i = 0; i < dimLabel; i++) {
					labelVec[i] = (label == i) ? 1.0 : 0.0;
				}

				Sample s = new Sample();
				s.data = Tensor.create(sampleDim);
				s.label = Tensor.create(labelVec);

				int value;
				for (int i = 0; i < channel; i++) {
					in.read(data);
					featureVec[i] = new double[dimRows * dimRows];
					for (int j = 0; j < featureVec[i].length; j++) {
						value = (int) data[j] & 0xff;
						if (normalize)
							featureVec[i][j] = (double) (value - 127) / 128.0;
						else
							featureVec[i][j] = (double) value;
					}
					INDArray channelData = Nd4j.create(featureVec[i]).reshape(dimRows, dimRows).transpose();
					s.data.putSlice(i, channelData);
				}
				samples.add(s);
			}
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (in != null) {
				try {
					in.close();
				} catch (IOException e) {}
			}
		}

		Sample[] arr = new Sample[samples.size()];
		arr = samples.toArray(arr);

		System.out.println(String.format("Loaded %d samples from %s", samples.size(), path));
		return arr;
	}
}