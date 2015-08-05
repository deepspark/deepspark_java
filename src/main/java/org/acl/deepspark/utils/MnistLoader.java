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
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.jblas.DoubleMatrix;
import org.nd4j.linalg.factory.Nd4j;

public class MnistLoader implements Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 4845357475294611873L;
	private static final int dimRows = 28;
	private static final int dimLabel = 10;

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
				int[] dimData = {1, dimRows, dimRows};
				
				s.data = Nd4j.zeros(dimData);
				s.data.setData(Nd4j.createBuffer(featureVec));
				s.data.transposei();
				s.data.divi(256);
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
				double label = Double.parseDouble(feature[dimRows * dimRows]);
				double[] featureVec = new double[dimRows * dimRows];
				double[] labelVec = new double[dimLabel];
				for (int i = 0; i < dimLabel; i++) {
					labelVec[i] = (label == i) ? 1.0 : 0.0;
				}

				for (int i = 0; i < feature.length -1; i++)
					featureVec[i] = Double.parseDouble(feature[i]);
				
				Sample s = new Sample();
				int[] dimData = {1, dimRows, dimRows};

				s.data = Nd4j.zeros(dimData);
				s.data.setData(Nd4j.createBuffer(featureVec));
				s.data.transposei();
				if (normalize)
					s.data.divi(256);
				s.label = Nd4j.zeros(dimLabel, 1);
				s.label.setData(Nd4j.createBuffer(labelVec));

				return s;
			}
		});
		return ret;
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

			while ((line = reader.readLine()) != null) {
				feature = line.split("\t");
				label = Double.parseDouble(feature[dimRows * dimRows]);
				double[] labelVec = new double[dimLabel];
				for (int i = 0; i < dimLabel; i++) {
					labelVec[i] = (label == i) ? 1.0 : 0.0;
				}

				for (int i = 0; i < feature.length -1; i++)
					featureVec[i] = Double.parseDouble(feature[i]);
				Sample s = new Sample();
				int[] dimData = {1, dimRows, dimRows};

				s.data = Nd4j.zeros(dimData);
				s.data.setData(Nd4j.createBuffer(featureVec));
				s.data.transposei();
				if (normalize)
					s.data.divi(256);
				s.label = Nd4j.zeros(dimLabel, 1);
				s.label.setData(Nd4j.createBuffer(labelVec));

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
