package org.acl.deepspark.nn.conf.spark;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.ListIterator;

import org.acl.deepspark.data.DeltaWeight;
import org.acl.deepspark.data.Sample;
import org.acl.deepspark.nn.layers.BaseLayer;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.jblas.DoubleMatrix;


public class DistNeuralNetConfiguration implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = -6624391825370570205L;
	
	private double learningRate;	
	private int epoch;
	private double momentum;
	private int minibatchSize;
	
	private List<BaseLayer> layerList;
	
	//running options	
	private boolean verbosity = true;
	private JavaSparkContext sc; 
	
	public DistNeuralNetConfiguration(double learningRate, int epoch, int minibatchSize, JavaSparkContext sc) {
		layerList = new ArrayList<BaseLayer>();
		this.epoch = epoch;
		this.minibatchSize = minibatchSize;
		this.sc = sc;
	}
	
	public DistNeuralNetConfiguration(double d, int i, int j, JavaSparkContext sc, boolean b) {
		this(d,i,j,sc);
		verbosity =b;
	}

	public void addLayer(BaseLayer l) {
		layerList.add(l);
	}
	
	public int getNumberOfLayers() {
		return layerList.size();
	}
	
	public void training(Sample[] data) {
		List<Sample> data_list = Arrays.asList(data);
		JavaRDD<Sample> rdd_data = sc.parallelize(data_list);
		int numMinibatch = (int) Math.ceil((double) data.length / minibatchSize); 
		double[] batchWeight = new double[numMinibatch];
		for(int i = 0; i < numMinibatch; i++)
			batchWeight[i] = 1.0 / numMinibatch;		
		JavaRDD<Sample>[] rdd_minibatch = rdd_data.randomSplit(batchWeight);
		
		for(int i = 0 ; i < epoch ; i++) {
			System.out.println(String.format("%d epoch...", i+1));
			// per epoch
			for(int j = 0; j < rdd_minibatch.length; j++) {
				if(verbosity)
					System.out.println(String.format("%d - epoch, %d minibatch",i+1, j / minibatchSize + 1));
				// per minibatch
				//get output
				JavaRDD<DoubleMatrix> delta = rdd_minibatch[j].map(new Function<Sample, DoubleMatrix>() {
					private static final long serialVersionUID = 7864025980071700556L;

					@Override
					public DoubleMatrix call(Sample arg0) throws Exception {
						return getOutput(arg0.data)[0].sub(arg0.label);
					}
				});
				
				
				//backpropagation
				JavaRDD<DeltaWeight> dWeight = delta.map(new Function<DoubleMatrix, DeltaWeight>() {
					/**
					 * 
					 */
					private static final long serialVersionUID = 7623058973015796084L;

					@Override
					public DeltaWeight call(DoubleMatrix arg0) throws Exception {
						DoubleMatrix[] error = new DoubleMatrix[0];
						error[0] = arg0;
						return backpropagate(error);
					}
				});
				
				// reduce weight
				DeltaWeight gradient = dWeight.reduce(new Function2<DeltaWeight, DeltaWeight, DeltaWeight>() {
					
					/**
					 * 
					 */
					private static final long serialVersionUID = 8486768340399892033L;

					@Override
					public DeltaWeight call(DeltaWeight arg0, DeltaWeight arg1)	throws Exception {
						DeltaWeight result = new DeltaWeight(arg0.gradBList.length);
											
						for(int i = 0; i < result.gradWList.length; i++) {
							result.gradWList[i] = new DoubleMatrix[arg0.gradWList[i].length][];
							for(int j = 0; j < result.gradWList[i].length; j++) {
								result.gradWList[i][j] = new DoubleMatrix[arg0.gradWList[i][j].length];
								for(int k = 0; k < result.gradWList[i][j].length;k++) {
									result.gradWList[i][j][j] = arg0.gradWList[i][j][k].add(arg1.gradWList[i][j][k]);
								}
							}	
						}
						
						for(int i = 0; i < result.gradBList.length; i++) {
							result.gradBList[i] = new double[arg0.gradBList[i].length];
							for(int j = 0; j < result.gradBList[i].length; j++) {
								result.gradBList[i][j] = arg0.gradBList[i][j] + arg1.gradBList[i][j];
							}	
						}
							
						return result;
					}
				});
				
				//update
				update(gradient);
			}
		}
	}
	
	private void update(DeltaWeight gradient) {
		ListIterator<BaseLayer> it = layerList.listIterator(getNumberOfLayers());
		int layerIdx = layerList.size();
		
		while(it.hasPrevious()) {
			layerIdx--;
			
			BaseLayer a = it.previous();
			DoubleMatrix[][] gradW = gradient.gradWList[layerIdx];
			double[] gradB = gradient.gradBList[layerIdx];
			
			if(gradW != null && gradB != null) {
				for(int i = 0; i < gradW.length ; i++)
					for(int j = 0; j < gradW[i].length ; j++)
						gradW[i][j].divi(minibatchSize);
				for(int i = 0; i < gradB.length ; i++)
					gradB[i] /= minibatchSize;
				
				a.update(gradient.gradWList[layerIdx], gradient.gradBList[layerIdx]);
			}
		}
	}
	
	
	public DeltaWeight backpropagate(DoubleMatrix[] delta) {
		DeltaWeight deltas = new DeltaWeight(layerList.size());
		ListIterator<BaseLayer> it = layerList.listIterator(getNumberOfLayers());
		int layerIdx = layerList.size();
		// Back-propagation
		while(it.hasPrevious()) {
			layerIdx--;
			
			BaseLayer a = it.previous();
			a.setDelta(delta);
			
			deltas.gradWList[layerIdx] = a.deriveGradientW();
			double[] b = new double[a.getDelta().length];
			for(int i = 0; i < a.getDelta().length; i++)
				b[i] = a.getDelta()[i].sum();
			deltas.gradBList[layerIdx] = b;
			
			delta = a.deriveDelta();
		}
		return deltas;
	}
	
	public DoubleMatrix[] getOutput(DoubleMatrix[] data) {
		Iterator<BaseLayer> itLayer = layerList.iterator();
		DoubleMatrix[] output = data;
		
		// Feed-forward
		while (itLayer.hasNext()) {
			BaseLayer l = itLayer.next();
			l.setInput(output);
			output = l.getOutput();
		}
		return output;
	}
}
