package org.acl.deepspark.nn.layers;

import org.acl.deepspark.data.Tensor;
import org.acl.deepspark.data.Weight;
import org.acl.deepspark.nn.conf.LayerConf;

public interface Layer {

	// initialization
	public abstract Weight createWeight(LayerConf conf, int[] input);
	public abstract int[] calculateOutputDimension(LayerConf conf, int[] input);
	
	// generate output	
	public abstract Tensor generateOutput(Weight weight, Tensor input);
	public abstract Tensor activate(Tensor output);
	
	// generate output for batch
//	public abstract Tensor generateOutputBatch(Weight weight, Tensor input);
	
	// backpropagation
	public abstract Tensor deriveDelta(Tensor activated, Tensor error);		// compute delta = f'(output) * error
	public abstract Weight gradient(Tensor input, Tensor error); 			// compute dJ/dw = input * delta
	public abstract Tensor calculateBackprop(Weight weight, Tensor error);  // compute backprop delta = transpose(w) * error
	
	// backpropagation for batch
//	public abstract Weight gradientBatch(Tensor input, Tensor error); 			// compute dJ/dw = input * delta
//	public abstract Tensor calculateBackpropBatch(Weight weight, Tensor error);  // compute backprop delta = transpose(w) * error
	
}
