package org.acl.deepspark.nn.layers;

import org.acl.deepspark.nn.conf.LayerConf;
import org.nd4j.linalg.api.ndarray.INDArray;

public interface Layer {

	public abstract INDArray createWeight(LayerConf conf, int[] input);
	public abstract INDArray generateOutput(INDArray weight, INDArray input);
	public abstract INDArray deriveDelta(INDArray weight, INDArray error);
	public abstract INDArray gradient(INDArray input, INDArray error);
}
