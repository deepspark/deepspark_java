package org.acl.deepspark.nn.layers;

import org.acl.deepspark.data.Weight;
import org.acl.deepspark.nn.conf.LayerConf;
import org.nd4j.linalg.api.ndarray.INDArray;

public interface Layer {

	public abstract Weight createWeight(LayerConf conf, int[] input);
	public abstract int[] calculateOutputDimension(LayerConf conf, int[] input);
	public abstract INDArray generateOutput(Weight weight, INDArray input);
	public abstract INDArray activate(INDArray output);
	
	public abstract INDArray deriveDelta(INDArray output, INDArray error);		// compute delta = f'(output) * error
	public abstract Weight gradient(INDArray input, INDArray error); 			// compute dJ/dw = input * delta
	public abstract INDArray calculateBackprop(Weight weight, INDArray error);  // compute backprop delta = w' * error
	
}
