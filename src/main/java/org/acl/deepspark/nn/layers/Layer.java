package org.acl.deepspark.nn.layers;

import org.acl.deepspark.data.Weight;
import org.nd4j.linalg.api.ndarray.INDArray;

public interface Layer {
	public abstract INDArray generateOutput(Weight w, INDArray input);
	public abstract INDArray deriveDelta(INDArray error);
	public abstract Weight gradient(INDArray error);
}
