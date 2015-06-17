package org.acl.deepspark.nn.conf.spark;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.ListIterator;

import org.acl.deepspark.data.DeltaAccumulator;
import org.acl.deepspark.data.DeltaWeight;
import org.acl.deepspark.data.Sample;
import org.acl.deepspark.nn.layers.BaseLayer;
import org.apache.spark.Accumulator;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.VoidFunction;
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
	
	//spark
	private transient JavaSparkContext sc = null;
	private Accumulator<DeltaWeight> accW = null;
	private boolean finalize = false; 
	
	public DeltaWeight getEmptyDeltaWeight() {
		if(!finalize)
			return null;
		
		DeltaWeight d = new DeltaWeight(layerList.size());
		Iterator<BaseLayer> itLayer = layerList.iterator();
		int count = 0;
		while (itLayer.hasNext()) {
			BaseLayer l = itLayer.next();
			int[] info = l.getWeightInfo();
			if(info == null) {
				d.gradWList[count] = null;
				d.gradBList[count] = null;
			} else {
				d.gradWList[count] = new DoubleMatrix[info[0]][info[1]];
				d.gradBList[count] = new double[info[0]];
				
				for(int i = 0; i < info[0]; i++) {
					for(int j = 0; j < info[j]; j++) {
						d.gradWList[count][i][j] = DoubleMatrix.zeros(info[2],info[3]);
					}
					d.gradBList[count][i] = 0;
				}
			}
			count++;
		}
		
		return d;
	}
	
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
		if(!finalize )
			return;
		
		List<Sample> data_list = Arrays.asList(data);
		JavaRDD<Sample> rdd_data = sc.parallelize(data_list).cache();
		int numMinibatch = (int) Math.ceil((double) data.length / minibatchSize); 
		double[] batchWeight = new double[numMinibatch];
		for(int i = 0; i < numMinibatch; i++)
			batchWeight[i] = 1.0 / numMinibatch;		
		JavaRDD<Sample>[] rdd_minibatch = rdd_data.randomSplit(batchWeight);
		accW = sc.accumulator(getEmptyDeltaWeight(), new DeltaAccumulator());
		
		for(int i = 0 ; i < epoch ; i++) {
			System.out.println(String.format("%d epoch...", i+1));
			// per epoch
			for(int j = 0; j < rdd_minibatch.length; j++) {
				if(verbosity)
					System.out.println(String.format("%d - epoch, %d minibatch",i+1, j + 1));
								
				// per minibatch
				//get output
				JavaRDD<DoubleMatrix> delta = rdd_minibatch[j].map(new OutputFunction(this)).cache();
				
				//backpropagation
				JavaRDD<DeltaWeight> dWeight = delta.map(new Function<DoubleMatrix, DeltaWeight>() {

					@Override
					public DeltaWeight call(DoubleMatrix arg0) throws Exception {
						DoubleMatrix[] error = new DoubleMatrix[1];
						error[0] = arg0;
						return backpropagate(error);
					}
				}).cache();
				
				accW.setValue(getEmptyDeltaWeight());
				
				// reduce weight
				dWeight.foreach(new VoidFunction<DeltaWeight>() {
					
					@Override
					public void call(DeltaWeight arg0) throws Exception {
						accW.add(arg0);
					}
				});
				 
				 DeltaWeight gradient = accW.value();
				//update
				update(gradient);
			}
		}
	}
	
	public void prepareForTraining(int[] dimIn) {
		finalize = true;
		Iterator<BaseLayer> itLayer = layerList.iterator();
		
		// Feed-forward
		while (itLayer.hasNext()) {
			BaseLayer l = itLayer.next();
			dimIn = l.initWeights(dimIn);
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
				
				a.update(gradW, gradB);
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
			
			if(a.getDelta() != null) {	
				double[] b = new double[a.getDelta().length];
				for(int i = 0; i < a.getDelta().length; i++)
					b[i] = a.getDelta()[i].sum();
				deltas.gradBList[layerIdx] = b;
			}			
			
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
