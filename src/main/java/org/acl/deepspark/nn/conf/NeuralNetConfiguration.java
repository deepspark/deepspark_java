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
	
	public NeuralNetConfiguration(double learningRate, double momentum) {
		// this.learningRate = learningRate;
		// this.momentum = momentum;
		layerList = new ArrayList<BaseLayer>();
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
		DoubleMatrix[] f = getOutput(data);
		DoubleMatrix[] delta = new DoubleMatrix[1];
		
		delta[0] = f[0].sub(label[0]);
		
		backpropagate(delta);
	}
	
	public void backpropagate(DoubleMatrix[] delta) {
		ListIterator<BaseLayer> it = layerList.listIterator(getNumberOfLayers());
		while(it.hasPrevious()) {
			BaseLayer a = it.previous();
			
			delta = a.update(delta);
		}
			
	}
	
	public DoubleMatrix[] getOutput(DoubleMatrix[] data) {
		Iterator<BaseLayer> itLayer = layerList.iterator();
		DoubleMatrix[] output = data;
		
		//feed-forward
		while(itLayer.hasNext()) {
			BaseLayer l = itLayer.next();
			l.setInput(output);
			output = l.getOutput();
		}
		return output;
	}
}
