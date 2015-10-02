package org.acl.deepspark.utils;

import java.io.*;
import java.util.ArrayList;

import org.acl.deepspark.data.Sample;
import org.acl.deepspark.data.Tensor;
import org.apache.commons.math.stat.descriptive.StatisticalSummary;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.jblas.FloatMatrix;

public class CIFARLoader implements Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 2453031569679527445L;
		
	private static final int dimRows = 32;
	private static final int dimCols = 32;
	private static final int channel = 3;
	private static final int dimLabel = 10;
			
	public static Sample[] loadIntoSamples(String path, boolean normalize) {
		int label;
		byte[] data = new byte[channel * dimRows * dimCols];
		final int[] sampleDim = new int[]{channel, dimRows, dimCols};
		ArrayList<Sample> samples = new ArrayList<Sample>();

		FileInputStream in = null;
		try {
			in = new FileInputStream(path);
			while ((label = in.read()) != -1) {
				float[] labelVec = new float [dimLabel];
				float[] featureVec = new float [channel * dimRows * dimCols];
				for (int i = 0; i < dimLabel; i++) {
					labelVec[i] = (label == i) ? 1 : 0;
				}

				int value;
				int length = featureVec.length;

				in.read(data);
				for (int i = 0 ; i < length; i++) {
					value = (int) data[i]&0xff;
					featureVec[i] = (float) value;
				}

				Sample s = new Sample();
				s.data = Tensor.create(featureVec, sampleDim);
				if (normalize) {
					s.data.subi(s.data.mean());
				}
				s.label = Tensor.create(labelVec, new int[] {dimLabel});
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
		byte[] data = new byte[channel * dimRows * dimCols];
		final int[] sampleDim = new int[]{channel, dimRows, dimCols};
		ArrayList<Sample> samples = new ArrayList<Sample>();

		FSDataInputStream in = null;
		try {
			Path p = new Path(path);
			FileSystem fs = FileSystem.get(new Configuration());
			in = fs.open(p);

			while ((label = in.read()) != -1) {

				float[] labelVec = new float [dimLabel];
				float[] featureVec = new float [channel * dimRows * dimCols];
				for (int i = 0; i < dimLabel; i++) {
					labelVec[i] = (label == i) ? 1 : 0;
				}

				int value;
				int length = featureVec.length;

				in.read(data);
				for (int i = 0 ; i < length; i++) {
					value = (int) data[i]&0xff;
					if (normalize)
						featureVec[i] = ((float) value - 128) / 128;
					else
						featureVec[i] = (float) value;
				}

				Sample s = new Sample();
				s.data = Tensor.create(featureVec, sampleDim);
				s.label = Tensor.create(labelVec, new int[] {dimLabel});
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