package org.acl.deepspark.nn.conf;

import java.util.ArrayList;
import java.util.List;

import org.acl.deepspark.nn.layers.BaseLayer;
import org.jblas.DoubleMatrix;


public class NeuralNetConfiguration {
	private List<BaseLayer> layerList;
	private int numLayer;
	private List<DoubleMatrix[]> outputList;
	private List<DoubleMatrix[]> deltaList;
	private double learningRate;
	
	public NeuralNetConfiguration(double learningRate) {
		this.learningRate = learningRate;
		layerList = new ArrayList<BaseLayer>();
		numLayer = 0;
	}
	
	public void addLayer(BaseLayer l) {
		layerList.add(l);
		numLayer++;
	}
	
	
}
