package org.acl.deepspark.nn.conf;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.ListIterator;

import org.acl.deepspark.data.Sample;
import org.acl.deepspark.nn.layers.BaseLayer;
import org.jblas.DoubleMatrix;


public class NeuralNetConfiguration implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = -2798664570482111852L;
	private double learningRate;	
	private int epoch;
	private double momentum;
	private int minibatchSize;
	
	private List<BaseLayer> layerList;
	private DoubleMatrix[][][] gradWList;
	private double[][] gradBList;
	
	//running options	
	private boolean verbosity = true; 
	private boolean finalize = false;
	
	public NeuralNetConfiguration(double learningRate, int epoch, int minibatchSize) {
		layerList = new ArrayList<BaseLayer>();
		this.epoch = epoch;
		this.minibatchSize = minibatchSize;
	}
	
	public NeuralNetConfiguration(double d, int i, int j, boolean b) {
		this(d,i,j);
		verbosity =b;
	}

	public void addLayer(BaseLayer l) {
		if(!finalize)
			layerList.add(l);
	}
	
	public int getNumberOfLayers() {
		return layerList.size();
	}
	
	public void training(Sample[] data) {
		if(!finalize)
			return;
		
		final DoubleMatrix[] delta = new DoubleMatrix[1];
				
		for(int i = 0 ; i < epoch ; i++) {
			System.out.println(String.format("%d epoch...", i+1));
			// per epoch
			for(int j = 0; j < data.length; j += minibatchSize) {
				if(verbosity)
					System.out.println(String.format("%d - epoch, %d minibatch",i+1, j / minibatchSize + 1));
				// per minibatch
				initGradList();

				int batchIter = Math.min(data.length, j+ minibatchSize);
				for(int k = j; k < batchIter; k++) {
					delta[0] = getOutput(data[k].data)[0].sub(data[k].label);
					backpropagate(delta);
				}
				
				update();
			}
		}
	}
	
	public void prepareForTraining(int dimInput) {
		finalize = true;
	}
	
	private void update() {
		ListIterator<BaseLayer> it = layerList.listIterator(getNumberOfLayers());
		int layerIdx = layerList.size();
		
		while(it.hasPrevious()) {
			layerIdx--;
			
			BaseLayer a = it.previous();
			DoubleMatrix[][] gradW = gradWList[layerIdx];
			double[] gradB = gradBList[layerIdx];
			
			if(gradW != null && gradB != null) {
				for(int i = 0; i < gradW.length ; i++)
					for(int j = 0; j < gradW[i].length ; j++)
						gradW[i][j].divi(minibatchSize);
				for(int i = 0; i < gradB.length ; i++)
					gradB[i] /= minibatchSize;
				
				a.update(gradWList[layerIdx], gradBList[layerIdx]);
			}
		}
	}
	
	private void initGradList() {
		gradWList = new DoubleMatrix[layerList.size()][][];
		gradBList = new double[layerList.size()][];
	}
	
	
	public void backpropagate(DoubleMatrix[] delta) {
		ListIterator<BaseLayer> it = layerList.listIterator(getNumberOfLayers());
		int layerIdx = layerList.size();
		// Back-propagation
		while(it.hasPrevious()) {
			layerIdx--;
			
			BaseLayer a = it.previous();
			a.setDelta(delta);
			
			if(gradWList[layerIdx] == null)
				gradWList[layerIdx] = a.deriveGradientW();
			else {
				DoubleMatrix[][] deltaW = a.deriveGradientW();
				for(int i = 0; i < gradWList[layerIdx].length; i++)
					for(int j = 0; j < gradWList[layerIdx][i].length ; j++)
						gradWList[layerIdx][i][j].addi(deltaW[i][j]);
			}
			
			
			
			if(gradBList[layerIdx] == null) {
				if(a.getDelta() != null) {			
					double[] b = new double[a.getDelta().length];
					for(int i = 0; i < a.getDelta().length; i++)
						b[i] = a.getDelta()[i].sum();
					gradBList[layerIdx] = b;
				}
			}
			else
				for(int i = 0; i < a.getDelta().length; i++)
					gradBList[layerIdx][i] += a.getDelta()[i].sum();
			
			delta = a.deriveDelta();
		}
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
