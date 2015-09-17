package org.acl.deepspark.utils;

import org.acl.deepspark.data.Sample;
import org.acl.deepspark.data.Tensor;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;

import java.io.*;
import java.util.ArrayList;

public class MnistLoader implements Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 4845357475294611873L;
	private static final int dimRows = 28;
	private static final int dimLabel = 10;

	public static Sample[] loadIntoSamples(String path, boolean normalize) {
		System.out.println("Data Loading...");
		float label;
		int[] dimData = {1, dimRows, dimRows};
		BufferedReader reader = null;
		ArrayList<Sample> samples = new ArrayList<Sample>();
		
		float[] featureVec = new float[dimRows * dimRows];
		try {
			reader = new BufferedReader(new FileReader(path));
			String line = null;
			String[] feature = null;
			while((line = reader.readLine()) != null) {
				feature = line.split("\t");
				float[] labelVec = new float[dimLabel];
				label = Float.parseFloat(feature[dimRows * dimRows]);
				for(int i = 0; i < dimLabel; i++) {
					labelVec[i] = (label == i) ?  1 : 0;
				}
				for(int i = 0; i < feature.length - 1;i++)
					featureVec[i] = Float.parseFloat(feature[i]);
				
				Sample s = new Sample();
				s.data = Tensor.create(featureVec, dimData).transpose();
				if (normalize)
					s.data.subi(128).divi(128);
				s.label = Tensor.create(labelVec, new int[] {dimLabel});

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
	
	public static JavaRDD<Sample> loadRDDFromHDFS(String path, final boolean normalize, JavaSparkContext sc) {
		JavaRDD<String> lines = sc.textFile(path);
		JavaRDD<Sample> ret = lines.map(new Function<String, Sample>() {

			/**
			 * 
			 */
			private static final long serialVersionUID = 1L;

			@Override
			public Sample call(String v1) throws Exception {
				String[] feature = v1.split("\t");
				float label = Float.parseFloat(feature[dimRows * dimRows]);
				float[] featureVec = new float[dimRows * dimRows];
				float[] labelVec = new float[dimLabel];
				for (int i = 0; i < dimLabel; i++) {
					labelVec[i] = (label == i) ? 1 : 0;
				}

				for (int i = 0; i < feature.length -1; i++)
					featureVec[i] = Float.parseFloat(feature[i]);
				
				Sample s = new Sample();
				int[] dimData = {1, dimRows, dimRows};

				s.data = Tensor.create(featureVec, dimData);
				if (normalize)
					s.data.subi(128).divi(128);
				s.label = Tensor.create(labelVec, new int[] {dimLabel});

				return s;
			}
		});
		return ret;
	}

	public static Sample[] loadFromHDFS(String path, boolean normalize) {
		System.out.println("Data Loading...");
		float label;
		int[] dimData = {1, dimRows, dimRows};
		BufferedReader reader = null;
		ArrayList<Sample> samples = new ArrayList<Sample>();
		float[] featureVec = new float[dimRows * dimRows];

		try {
			Path p = new Path(path);
			FileSystem fs = FileSystem.get(new Configuration());
			reader = new BufferedReader(new InputStreamReader(fs.open(p)));
			String line = null;
			String[] feature = null;

			while ((line = reader.readLine()) != null) {
				feature = line.split("\t");
				float[] labelVec = new float[dimLabel];
				label = Float.parseFloat(feature[dimRows * dimRows]);
				for (int i = 0; i < dimLabel; i++) {
					labelVec[i] = (label == i) ? 1 : 0;
				}

				for (int i = 0; i < feature.length -1; i++)
					featureVec[i] = Float.parseFloat(feature[i]);

				Sample s = new Sample();
				s.data = Tensor.create(featureVec, dimData).transpose();
				if (normalize)
					s.data.subi(128).divi(128);
				s.label = Tensor.create(labelVec, new int[] {dimLabel});

				samples.add(s);
			}

		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (reader != null) {
				try {
					reader.close();
				} catch (IOException e) {}
			}
		}

		Sample[] arr = new Sample[samples.size()];
		arr = samples.toArray(arr);

		System.out.println(String.format("Loaded %d samples from %s", samples.size(), path));
		return arr;
	}


}
