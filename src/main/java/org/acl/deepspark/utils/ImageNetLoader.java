package org.acl.deepspark.utils;

public class ImageNetLoader {
	private static final long serialVersionUID = 4845357475294611873L;
	
/*	private static final int dimRows = 28;
	private static final int dimChannels = 3;
	private static final int dimLabel = 1000;
	
	public static Sample[] loadFromHDFS(String path, boolean normalize) {
		BufferedReader reader = null;
		double label;
		ArrayList<Sample> samples = new ArrayList<Sample>();
		double[][] featureVec = new double[dimChannels][dimRows * dimRows];
		
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
				
				for(int j = 0; j < feature.length -1;j++) {
					for (int i = 0; i < dimChannels; i++) {
						featureVec[i][j] = Double.parseDouble(feature[j].split(" ")[i]);
					}
				}
				
				Sample s = new Sample();
				DoubleMatrix[] sample = new DoubleMatrix[dimChannels];
				for (int i = 0; i < dimChannels; i++) {
					sample[i] = new DoubleMatrix(dimRows, dimRows, featureVec[i].clone()).transpose();
				}
				
				if(normalize) {
					for (int i = 0; i < dimChannels; i++)
						sample[i].divi(256);
				}
				
				s.data = sample;
				s.label = sampleLabel;
				
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
		double[][] featureVec = new double[dimChannels][dimRows * dimRows];
		
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
				
				for(int j = 0; j < feature.length -1;j++) {
					for (int i = 0; i < dimChannels; i++) {
						featureVec[i][j] = Double.parseDouble(feature[j].split(" ")[i]);
					}
				}
				
				Sample s = new Sample();
				DoubleMatrix[] sample = new DoubleMatrix[dimChannels];
				for (int i = 0; i < dimChannels; i++) {
					sample[i] = new DoubleMatrix(dimRows, dimRows, featureVec[i].clone()).transpose();
				}
				
				if(normalize) {
					for (int i = 0; i < dimChannels; i++)
						sample[i].divi(256);
				}
				
				s.data = sample;
				s.label = sampleLabel;
				
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
*/
}
