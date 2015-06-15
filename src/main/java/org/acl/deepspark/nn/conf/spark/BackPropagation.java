package org.acl.deepspark.nn.conf.spark;

import org.acl.deepspark.data.DeltaWeight;
import org.apache.spark.api.java.function.Function;
import org.jblas.DoubleMatrix;

public class BackPropagation implements Function<DoubleMatrix, DeltaWeight> {
	/**
	 * 
	 */
	private DistNeuralNetConfiguration nnc;
	
	public BackPropagation(DistNeuralNetConfiguration nnc) {
		this.nnc = nnc;
	}
	private static final long serialVersionUID = 4178848926738610617L;

	@Override
	public DeltaWeight call(DoubleMatrix arg0) throws Exception {
		DoubleMatrix[] error = new DoubleMatrix[0];
		error[0] = arg0;
		return nnc.backpropagate(error);
	}

}
