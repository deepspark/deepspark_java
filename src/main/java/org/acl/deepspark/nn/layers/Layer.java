package org.acl.deepspark.nn.layers;

import org.acl.deepspark.data.Tensor;
import org.acl.deepspark.data.Weight;
import org.acl.deepspark.nn.conf.LayerConf;

public interface Layer {

	// initialization
	public abstract Weight	createWeight(LayerConf conf, int[] input);
	public abstract int[]	calcOutputShape();
	
	// feedForward
	public abstract Tensor	generateOutput(Weight weight, Tensor input);
	public abstract Tensor	activate(Tensor output);

	// backPropagation
	public abstract Tensor	deriveDelta(Tensor activated, Tensor error);		// compute delta = f'(output) * error
	public abstract Weight	gradient(Tensor input, Tensor error); 				// compute dJ/dw = input * delta
	public abstract Tensor	calculateBackprop(Weight weight, Tensor error);  	// compute backprop delta = transpose(w) * error

}
