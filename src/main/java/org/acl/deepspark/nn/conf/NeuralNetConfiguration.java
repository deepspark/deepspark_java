package org.acl.deepspark.nn.conf;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.ListIterator;

import org.acl.deepspark.nn.layers.BaseLayer;
import org.jblas.DoubleMatrix;


public class NeuralNetConfiguration {
	private List<BaseLayer> layerList;
	
	/*
	private List<DoubleMatrix[]> outputList;
	private List<DoubleMatrix[]> deltaList;
	private double learningRate;
	private double momentum;
	*/
	
	private int epoch;
	
	public NeuralNetConfiguration(int epoch) {
		// this.learningRate = learningRate;
		// this.momentum = momentum;
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
		// outputList = new ArrayList<DoubleMatrix[]>();
		// deltaList = new ArrayList<DoubleMatrix[]>();
		
		//feed forward
		DoubleMatrix[] f;
		final DoubleMatrix[] sample = new DoubleMatrix[1];
		sample[0] = data[0];
		
		for(int i = 0 ; i < epoch ; i++) {
			System.out.println("epoch " + String.valueOf(i));
			f = getOutput(sample);
			DoubleMatrix[] delta = new DoubleMatrix[1];
			delta[0] = f[0].sub(label[0]);
			backpropagate(delta);
		}
	}
	
	public void backpropagate(DoubleMatrix[] delta) {
		System.out.println("backprop start");
		ListIterator<BaseLayer> it = layerList.listIterator(getNumberOfLayers());
		while(it.hasPrevious()) {
			BaseLayer a = it.previous();
			delta = a.update(delta);
		}
		System.out.println("backprop end");
	}
	
	public DoubleMatrix[] getOutput(DoubleMatrix[] data) {
		System.out.println("feedforward start");
		Iterator<BaseLayer> itLayer = layerList.iterator();
		DoubleMatrix[] output = data;
		
		//feed-forward
		while (itLayer.hasNext()) {
			BaseLayer l = itLayer.next();
			l.setInput(output);
			output = l.getOutput();
		}
		System.out.println("feedforward end");
		return output;
	}
}
