package org.acl.deepspark.nn.conf;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.ListIterator;

import org.acl.deepspark.nn.layers.BaseLayer;
import org.jblas.DoubleMatrix;
import org.jblas.util.Random;


public class NeuralNetConfiguration {
	private double learningRate;	
	private int epoch;
	private double momentum;
	
	private List<BaseLayer> layerList;
	
	public NeuralNetConfiguration(double learningRate, int epoch) {
		layerList = new ArrayList<BaseLayer>();
		this.epoch = epoch;
	}
	
	public void addLayer(BaseLayer l) {
		layerList.add(l);
	}
	
	public int getNumberOfLayers() {
		return layerList.size();
	}
	
	public void training(DoubleMatrix[] data, DoubleMatrix[] label) {
		if (data.length != label.length) {
			System.err.println("Mismatch of the number of data and labels");
			return;
		}
		
		final DoubleMatrix[] delta = new DoubleMatrix[1];
		final DoubleMatrix[] sample = new DoubleMatrix[1]; 
		int size = data.length;
		
		int sampleIdx;
		for(int i = 0 ; i < epoch ; i++) {
			sampleIdx = Random.nextInt(size);
			//System.out.println("epoch " + String.valueOf(i));
			sample[0] = data[sampleIdx];
			delta[0] = getOutput(sample)[0].sub(label[sampleIdx]);
			backpropagate(delta);
		}
	}
	
	public void backpropagate(DoubleMatrix[] delta) {
		ListIterator<BaseLayer> it = layerList.listIterator(getNumberOfLayers());
		// Back-propagation
		while(it.hasPrevious()) {
			BaseLayer a = it.previous();
			delta = a.update(delta);
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
	
	public static class Builder {
		private double learningRate;	
		private int epoch;
		private double momentum;
		
		public void learningRate(double learningRates) {
			this.learningRate = learningRates;
		}
		
		public void epoch(int epoch) {
			this.epoch = epoch;
		}
		
		public void momentum(double momentum) {
			this.momentum = momentum;
		}
		
		public void addLayer(BaseLayer l) {
			
		}
	}
}
