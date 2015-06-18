package org.acl.deepspark.nn.conf.spark;

import java.io.Serializable;
import java.util.List;

import org.acl.deepspark.data.DeltaWeight;
import org.acl.deepspark.data.Sample;
import org.acl.deepspark.nn.layers.BaseLayer;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.broadcast.Broadcast;
import org.jblas.DoubleMatrix;


public class DistNeuralNetConfiguration implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = -6624391825370570205L;
	
	private BaseLayer[] layerList;
	
	//spark
	private boolean finalize = false; 
	
	public DeltaWeight getEmptyDeltaWeight() {
		if(!finalize)
			return null;
		
		DeltaWeight d = new DeltaWeight(layerList.length);
		for(int count =0; count < layerList.length; count++) {
			BaseLayer l = layerList[count];
			int[] info = l.getWeightInfo();
			if(info == null) {
				d.gradWList[count] = null;
				d.gradBList[count] = null;
			} else {
				d.gradWList[count] = new DoubleMatrix[info[0]][info[1]];
				d.gradBList[count] = new double[info[0]];
				
				for(int i = 0; i < info[0]; i++) {
					for(int j = 0; j < info[1]; j++) {
						d.gradWList[count][i][j] = DoubleMatrix.zeros(info[2],info[3]);
					}
					d.gradBList[count][i] = 0;
				}
			}
		}
		
		return d;
	}
	
	public int getNumberOfLayers() {
		return layerList.length;
	}
	
	public void training(JavaRDD<Sample> data,JavaSparkContext sc) {
		if(!finalize )
			return;
		
		final Broadcast<BaseLayer[]> layers = sc.broadcast(layerList);
		
		// getting output;
		JavaRDD<DoubleMatrix> delta = data.map(new Function<Sample, DoubleMatrix>() {
			@Override
			public DoubleMatrix call(Sample v1) throws Exception {
				DoubleMatrix[] output = v1.data;
				for(int l = 0; l < layers.value().length; l++) {
					BaseLayer a = layers.value()[l];
					a.setInput(output);
					output = a.getOutput();
				}
				DoubleMatrix error = output[0].sub(v1.label);
				System.out.println(error.sum() / error.length);
				
				return error;
			}
		});
		
		//backpropagation
		JavaRDD<DeltaWeight> dWeight = delta.map(new Function<DoubleMatrix, DeltaWeight>() {
			@Override
			public DeltaWeight call(DoubleMatrix arg0) throws Exception {
				BaseLayer[] layerList = layers.value();
				DoubleMatrix[] error = new DoubleMatrix[1];
				error[0] = arg0;
				
				DeltaWeight deltas = new DeltaWeight(layerList.length);
				
				// Back-propagation
				for(int l = layerList.length - 1; l >=0 ; l--) {
					BaseLayer a =layerList[l];
					a.setDelta(error);
					
					deltas.gradWList[l] = a.deriveGradientW();
					
					if(a.getDelta() != null) {	
						double[] b = new double[a.getDelta().length];
						for(int i = 0; i < a.getDelta().length; i++)
							b[i] = a.getDelta()[i].sum();
						deltas.gradBList[l] = b;
					}			
					
					error = a.deriveDelta();
				}
				
				return deltas;
			}
		});
		
		
		
		DeltaWeight gradient = dWeight.fold(getEmptyDeltaWeight(), new Function2<DeltaWeight, DeltaWeight, DeltaWeight>() {
			@Override
			public DeltaWeight call(DeltaWeight v1, DeltaWeight v2) throws Exception {
				for(int i1 = 0; i1 < v1.gradWList.length; i1++) {
					if(v1.gradWList[i1] != null && v2.gradWList[i1] != null) {
						for(int j1 = 0; j1 < v1.gradWList[i1].length; j1++) {
							for(int k = 0; k < v1.gradWList[i1][j1].length;k++) {
								v1.gradWList[i1][j1][k].addi(v2.gradWList[i1][j1][k]);
							}
						}
						
						for(int j1 = 0; j1 < v1.gradBList[i1].length; j1++) {
							v1.gradBList[i1][j1] += v2.gradBList[i1][j1];
						}	
					}
				}
				
				return v1;
			}
		});				
		
		//update
		long minibatchSize = dWeight.count();
		update(gradient, (int) minibatchSize);
	}
	
	public void prepareForTraining(List<BaseLayer> l, int[] dimIn) {
		finalize = true;
		layerList = new BaseLayer[l.size()];
		layerList = l.toArray(layerList);
		
		for(int i = 0; i < 0; i++) {
			dimIn = layerList[i].initWeights(dimIn);
		}
	}
	
	private void update(DeltaWeight gradient,int minibatchSize) {
		for(int i = layerList.length - 1; i >=0 ; i--) {
			BaseLayer a = layerList[i];
			DoubleMatrix[][] gradW = gradient.gradWList[i];
			double[] gradB = gradient.gradBList[i];
			
			if(gradW != null && gradB != null) {
				for(int i1 = 0; i1 < gradW.length ; i1++)
					for(int j = 0; j < gradW[i1].length ; j++)
						gradW[i1][j].divi(minibatchSize);
				for(int i1 = 0; i1 < gradB.length ; i1++)
					gradB[i1] /= minibatchSize;
				
				a.update(gradW, gradB);
			}
		}
	}
	
	
	public DoubleMatrix[] getOutput(DoubleMatrix[] data) {
		DoubleMatrix[] output = data;
		for(int l = 0; l < layerList.length; l++) {
			BaseLayer a = layerList[l];
			a.setInput(output);
			output = a.getOutput();
		}
		return output;
	}
}
