package org.acl.deepspark.nn.layers;

import org.acl.deepspark.data.Weight;
import org.acl.deepspark.nn.conf.LayerConf;
import org.nd4j.linalg.api.ndarray.INDArray;

public interface Layer {

	public abstract Weight createWeight(LayerConf conf, int[] input);
	public abstract int[] calculateOutputDimension(LayerConf conf, int[] input);
	public abstract INDArray generateOutput(Weight weight, INDArray input);
	public abstract INDArray activate(INDArray output);
	public abstract Weight gradient(INDArray input, INDArray error);
	public abstract INDArray deriveDelta(Weight weight, INDArray error, INDArray output);
}
